from datetime import datetime
import logging
import os
import json
import torch
import asyncio
from transformers import (
    pipeline,
    BitsAndBytesConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer
)

from mlw_core import exceptions as exc
from mlw_models.message import (MessageMLTaskType,
                            MessageMLTaskModel,
                            MessageMLTaskStatus,
                            MessageMLTaskPromptRequest,
                            MessageMLTaskPromptResponseCompleted,
                            MessageMLTaskPromptResponseFailed,
                            MessageMLTaskPromptResponse,)


logger = logging.getLogger(__name__)

#-- Инициализация LLM
MODEL_ID = os.getenv("MODEL_ID", "HuggingFaceTB/SmolLM2-1.7B-Instruct")
QUANTIZATION_BITS = os.getenv("QUANTIZATION_BITS", "4")
logger.info(f"--- Loading ML Model: {MODEL_ID} with {QUANTIZATION_BITS}-bit quantization")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# Настройка защиты промтов
prompt_protection = os.getenv("PROMPT_PROTECTION", "off")
logger.info(f"INIT: prompt_protection: {prompt_protection}")
answer_text_case_attack = "Прошу прощения, я не могу ответить на этот вопрос."


# Инициализация пайплайна
generator = pipeline(
    "text-generation",
    model=MODEL_ID,
    model_kwargs={"quantization_config": bnb_config},
    tokenizer=MODEL_ID,
    device_map="auto"
)

#-- Инициализация классификатора промт-атак (ModernBERT)
llm_guard_model = None
llm_guard_tokenizer = None

if prompt_protection == "llm_guard":
    LLM_GUARD_PATH = os.getenv("LLM_GUARD_MODEL_PATH", "./_ml_pipeline/models/target-base")
    logger.info(f"--- Loading LLM Guard Model (ModernBERT) from: {LLM_GUARD_PATH} ---")

    try:
        llm_guard_tokenizer = AutoTokenizer.from_pretrained(LLM_GUARD_PATH)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Загружаем модель и сразу переводим в fp16 (.half()) для экономии VRAM
        llm_guard_model = AutoModelForSequenceClassification.from_pretrained(
            LLM_GUARD_PATH,
            num_labels=2
        ).to(device)

        if device == "cuda":
            llm_guard_model = llm_guard_model.half()

        llm_guard_model.eval()  # Переводим в строгий режим инференса (отключаем Dropout/BatchNorm)
        logger.info(f"--- LLM Guard ModernBERT successfully loaded into {device.upper()} VRAM! ---")
    except Exception as e:
        logger.critical(f"Failed to load LLM Guard Model: {e}. Falling back to 'off' mode.")
        prompt_protection = "off"



def _warmup_llm_model():
    logger.info("---Starting ML Model warmup (pre-loading weights into GPU---")
    try:
        warmup_messages = [{"role": "user", "content": "Hello"}]
        generator(
            warmup_messages,
            max_new_tokens=5,
            pad_token_id=generator.tokenizer.eos_token_id,
        )
        logger.info(f"--- ML Model warmup completed successfully! GPU is ready ---")
    except Exception as e:
        logger.error(f"--- ML Model warmup failed: {e} ---")


def _warmup_llm_guard_model():
    """
    Предварительный прогрев классификатора защитного шлюза (ModernBERT).
    """
    # Проверяем, включен ли режим в .env и инициализирована ли модель
    if prompt_protection == "llm_guard" and llm_guard_model is not None:
        logger.info("--- Starting LLM Guard ModernBERT warmup ---")
        try:
            prompt_protection_llm_guard("Warmup check for classifier initialization")
            logger.info("--- LLM Guard ModernBERT warmup completed successfully! ---")
        except Exception as e:
            logger.error(f"--- LLM Guard ModernBERT warmup failed: {e} ---")


def prompt_protection_basic(prompt_input: str) -> bool:
    forbidden_words = [
        "вирус",
        "взломай",
        "взломать",
    ]
    cleaned_input = prompt_input.lower()
    attack_detected = any(word in cleaned_input for word in forbidden_words)
    logger.info(f"DEBUG: attack_detected: {attack_detected}, prompt_input: {prompt_input}")
    return attack_detected


def prompt_protection_llm_guard(prompt_input: str) -> bool:
    """
    Классифицирует входящий промт с помощью обученного ModernBERT.
    Возвращает True, если обнаружена попытка джейлбрейка/атаки.
    """
    if not prompt_input or llm_guard_model is None or llm_guard_tokenizer is None:
        return False

    try:
        # 1. Токенизация (обрезаем до 512 токенов, как училась модель)
        inputs = llm_guard_tokenizer(
            str(prompt_input),
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        # 2. Отправляем тензоры на ту же видеокарту, где лежит модель
        inputs = {key: val.to(llm_guard_model.device) for key, val in inputs.items()}

        # 3. Быстрый инференс без расчета градиентов
        with torch.no_grad():
            outputs = llm_guard_model(**inputs)
            logits = outputs.logits

        # 4. Получаем класс с максимальной вероятностью (0 = Safe, 1 = Attack)
        pred_class = torch.argmax(logits, dim=-1).item()
        attack_detected = (pred_class == 1)

        logger.info(
            f"DEBUG [llm_guard]: pred_class={pred_class} (Attack={attack_detected}), prompt='{str(prompt_input)[:50]}...'")
        return attack_detected

    except Exception as e:
        logger.error(f"LLM Guard ModernBERT inference error: {e}")
        # В критических системах безопасности при сбое фильтра запрос лучше заблокировать (Fail-Safe)
        return True


def _generate_text_sync(prompt_input: str) -> str:
    """
    Synchronous CPU-blocking function for ML inference
    Args:
        prompt_input: User input for prompt
    Returns:
        Model response
    """
    messages = [
        {"role": "system",
         "content": "Ты - лаконичный и точный ассистент. Отвечай по существу, без лишних вступлений, приветствий, рассуждений, но при этом вежливым и дружелюбным тоном."},
        {"role": "user",
         "content": f"{prompt_input}"}
    ]

    output = generator(
        messages,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        pad_token_id=generator.tokenizer.eos_token_id
    )
    return output[0]['generated_text'][-1]['content']

async def process_ml_task(request_body: dict) -> MessageMLTaskPromptResponse:
    try:
        request_body_validated = MessageMLTaskPromptRequest.model_validate(request_body)
        prompt_input = request_body_validated.prompt_input

        # Проверка настроек режима защиты промт-атак
        attack_detected = False
        logger.info(f"DEBUG: prompt_protection: {prompt_protection}")

        if prompt_protection == "basic":
            logger.info(f"DEBUG: starting protection: {prompt_protection}")
            attack_detected = prompt_protection_basic(prompt_input)
        elif prompt_protection == "llm_guard":
            logger.info(f"DEBUG: starting ModernBERT protection: {prompt_protection}")
            # ВАЖНО: Оборачиваем синхронный PyTorch-инференс в to_thread,
            # чтобы не заблокировать асинхронный event loop RabbitMQ!
            attack_detected = await asyncio.to_thread(prompt_protection_llm_guard, prompt_input)


        if not attack_detected:
            prompt_output = await asyncio.to_thread(_generate_text_sync, prompt_input)
            logger.info(f"DEBUG3: prompt_input: {prompt_input}; prompt_output: {prompt_output}")
        else:
            prompt_output = answer_text_case_attack
            logger.warning(f"ATTACK DETECTED: {prompt_input}")
            logger.info(f"DEBUG: prompt_input: {prompt_input}; prompt_output: {prompt_output}")

        response_payload = MessageMLTaskPromptResponse(
            task_id=request_body_validated.task_id,
            task_created_at=request_body_validated.task_created_at,
            type=request_body_validated.type,
            model=request_body_validated.model,
            response=MessageMLTaskPromptResponseCompleted(
                status=MessageMLTaskStatus.COMPLETED,
                prompt_output=prompt_output
            )
        )
        return response_payload

    # except Exception as e:
    #     fallback_task_id = request_body_validated.get("task_id", "unknown-uuid")
    #     logger.error(f"ML Task {fallback_task_id} failed: {e}")
    # return MessageMLTaskPromptResponse(
    #         task_id=request_body_validated.task_id,
    #         task_created_at=request_body_validated.task_created_at,
    #         type=request_body_validated.type,
    #         model=request_body_validated.model,
    #         response=MessageMLTaskPromptResponseFailed(
    #             error_code="failed_to_process_task",
    #             error_message="Failed to process task",
    #         )
    #     )


    except Exception as e:
            # 1. Безопасно извлекаем объект валидации, если он успел создаться до ошибки
            validated = locals().get("request_body_validated", None)

            # 2. Собираем параметры для базового конструктора ответа.
            # Если валидация Pydantic упала на первой строчке, берем данные напрямую из сырого словаря request_body
            fallback_task_id = getattr(validated, "task_id", request_body.get("task_id", "00000000-0000-0000-0000-000000000000"))
            fallback_created_at = getattr(validated, "task_created_at", request_body.get("task_created_at", datetime.utcnow()))
            fallback_type = getattr(validated, "type", MessageMLTaskType.PROMPT)
            fallback_model = getattr(validated, "model", MessageMLTaskModel.DEMO_MODEL)

            logger.error(f"ML Task {fallback_task_id} failed with error: {e}", exc_info=True)

            # 3. Формируем строго валидный ответ согласно схеме MessageMLTaskPromptResponse
            return MessageMLTaskPromptResponse(
                task_id=fallback_task_id,
                task_created_at=fallback_created_at,
                type=fallback_type,
                model=fallback_model,
                response=MessageMLTaskPromptResponseFailed(
                    error_code="failed_to_process_task",
                    error_message=f"Failed to process task: {str(e)}",
                )
            )