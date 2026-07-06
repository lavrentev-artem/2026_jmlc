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
logger.info(f"--- Loading ML Model: {MODEL_ID}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# Pipeline initialization
generator = pipeline(
    "text-generation",
    model=MODEL_ID,
    model_kwargs={"quantization_config": bnb_config},
    device_map="auto"
)


def _generate_text_sync(prompt_input: str) -> str:
    """
    Synchronous CPU-blocking function for ML inference
    Args:
        prompt_input: User input for prompt
    Returns:
        Model response
    """
    messages = [
        {"role": "user",
         "content": f"I like these video games: {prompt_input}. Recommend me 1 new game according to my preferences. In your answer type only the game's title, do not type anything else."}
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
    request_body_validated = MessageMLTaskPromptRequest.model_validate(request_body)
    prompt_input = request_body_validated.prompt_input

    try:
        prompt_output = await asyncio.to_thread(_generate_text_sync, prompt_input)
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
        logger.error(f"ML Task {request_body_validated.task_id} failed: {e}")
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


# def process_ml_task(ch, method, properties, body):
#     try:
#         data = json.loads(body)
#         user

async def process_ml_task_mock(request_body: dict) -> MessageMLTaskPromptResponse:
    """
    Process ML Task
    Args:
        request_body: Task request body
    Returns:
        payload: Initial payload with added prompt_output
    """
    request_body_validated = MessageMLTaskPromptRequest.model_validate(request_body)
    prompt_input = request_body_validated.prompt_input

    try:
        prompt_output = "433"
        if request_body_validated.prompt_input == "115":
            prompt_output = None
        logger.info(f"DEBUG: input: {request_body_validated}; output: {prompt_output}")

        if prompt_output:
            response_payload = MessageMLTaskPromptResponse(
                task_id=request_body_validated.task_id,
                task_created_at=request_body_validated.task_created_at,
                type=request_body_validated.type,
                model=request_body_validated.model,
                response=MessageMLTaskPromptResponseCompleted(
                    status = MessageMLTaskStatus.COMPLETED,
                    prompt_output=prompt_output
                )
            )
            logger.info(f"DEBUG: response_payload: {response_payload}")
            return response_payload
        else:
            raise exc.MLTaskProcessingFailed


    except Exception as e:
        response_payload = MessageMLTaskPromptResponse(
            task_id=request_body_validated.task_id,
            task_created_at=request_body_validated.task_created_at,
            type=request_body_validated.type,
            model=request_body_validated.model,
            response=MessageMLTaskPromptResponseFailed(
                error_code="failed_to_process_task",
                error_message="Failed to process task",
            )
        )
        logger.error(f"ML Task {request_body['task_id']} failed with error: {e}")
        return response_payload
