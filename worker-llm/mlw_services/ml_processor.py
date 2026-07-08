import logging
import os
import json
import torch
import asyncio
from transformers import pipeline, BitsAndBytesConfig

from mlw_core import exceptions as exc
from mlw_models.message import (MessageMLTaskType,
                            MessageMLTaskModel,
                            MessageMLTaskStatus,
                            MessageMLTaskPromptRequest,
                            MessageMLTaskPromptResponseCompleted,
                            MessageMLTaskPromptResponseFailed,
                            MessageMLTaskPromptResponse,)


logger = logging.getLogger(__name__)

#-- ML model initialization
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


# Pipeline initialization
generator = pipeline(
    "text-generation",
    model=MODEL_ID,
    model_kwargs={"quantization_config": bnb_config},
    device_map="auto"
)
def _warmup_model():
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

    except Exception as e:
        fallback_task_id = request_body_validated.get("task_id", "unknown-uuid")
        logger.error(f"ML Task {fallback_task_id} failed: {e}")
    return MessageMLTaskPromptResponse(
            task_id=request_body_validated.task_id,
            task_created_at=request_body_validated.task_created_at,
            type=request_body_validated.type,
            model=request_body_validated.model,
            response=MessageMLTaskPromptResponseFailed(
                error_code="failed_to_process_task",
                error_message="Failed to process task",
            )
        )
