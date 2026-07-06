import core.exceptions as exc
from datetime import datetime
from uuid import UUID, uuid4
import logging

from models import MessageMLTaskPromptRequest, MessageMLTaskType, MessageMLTaskModel, MessageMLTaskStatus, MessageMLTaskPromptResponse
from services.adapters.rabbitmq_client import RabbitMQClient


logger = logging.getLogger(__name__)


async def prompt_ml_service(task_id: UUID,
                            task_created_at: datetime,
                            prompt_input: str,
                            mq_client: RabbitMQClient) -> dict:
    """
    Prompt ML service
    Args:
        prompt_input[str]: User's prompt
    Returns:
        prompt_output[str]: ML service response
    """
    try:
        # Prepare and validate message payload
        message_payload = MessageMLTaskPromptRequest (
            task_id=task_id,
            task_created_at=task_created_at,
            type=MessageMLTaskType.PROMPT,
            model=MessageMLTaskModel.DEMO_MODEL,
            prompt_input=prompt_input,
        )

        prompt_response = await mq_client.publish_message(
            payload=message_payload,
            routing_key="predict"
        )
        logger.info(f"ML Prompt Response - prompt_response: {prompt_response}")

        prompt_response_validated = MessageMLTaskPromptResponse.model_validate(prompt_response)
        logger.info(f"ML Prompt Response - prompt_response_validated: {prompt_response_validated}")

        if prompt_response_validated.response.status == MessageMLTaskStatus.COMPLETED:
            prompt_result = {
                "status": prompt_response_validated.response.status,
                "prompt_output": prompt_response_validated.response.prompt_output
            }
            return prompt_result

        elif prompt_response_validated.response.status == MessageMLTaskStatus.FAILED:
            logger.error(f"ML Worker failed to process task {prompt_response_validated.task_id}; error code: {prompt_response_validated.response.error_code}, error message: {prompt_response_validated.response.error_message}")
            prompt_result = {
                "status": prompt_response_validated.response.status,
                "error_code": prompt_response_validated.response.error_code,
                "error_message": prompt_response_validated.response.error_message
            }
            return prompt_result
        else:
            logger.error(f"ML Worker unknown error")
            raise exc.MLPromptFailed()


    except Exception as e:
        raise exc.MLPromptSendingFailed()


def mock_prompt_ml_service(prompt_input: str) -> str:
    """
    Prompt ML service
    Args:
        prompt_input[str]: User's prompt
    Returns:
        prompt_output[str]: ML service response
    """
    # MVP implementation: static response (will be refactored later)
    print(f"User's prompt: {prompt_input}")
    if prompt_input.lower() == "dishonored":
        prompt_output = "Dishonored 2"
    elif prompt_input.lower() == "dishonored, dishonored 2":
        prompt_output = "Deus Ex: Human Revolution"
    elif prompt_input.lower() == "deus ex: human devolution":
        prompt_output = "Deus Ex: Mankind Divided"
    elif prompt_input.lower() == "half life 2":
        prompt_output = "Prey"
    elif prompt_input.lower() == "half life 2, doom 3":
        prompt_output = "Far Cry"
    else:
        prompt_output = "42"

    if not prompt_output:
        raise exc.MLPromptFailed()
    print(f"Prompt output: {prompt_output}")
    return prompt_output

