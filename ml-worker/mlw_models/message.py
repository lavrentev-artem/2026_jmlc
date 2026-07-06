from enum import Enum
from typing import Optional, Literal, Union
from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID


#-- ENUMs
class MessageMLTaskType(str, Enum):
    PROMPT = "prompt"

class MessageMLTaskModel(str, Enum):
    DEMO_MODEL = "demo_model"

class MessageMLTaskStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"


#-- Models
#-- Message for ML Task of type PROMPT
class MessageMLTaskPromptBase(BaseModel):
    task_id: UUID
    task_created_at: datetime
    type: Literal[MessageMLTaskType.PROMPT] = MessageMLTaskType.PROMPT
    model: Literal[MessageMLTaskModel.DEMO_MODEL] = MessageMLTaskModel.DEMO_MODEL

class MessageMLTaskPromptRequest(MessageMLTaskPromptBase):
    prompt_input: str

class MessageMLTaskPromptResponseCompleted(BaseModel):
    status: Literal[MessageMLTaskStatus.COMPLETED] = MessageMLTaskStatus.COMPLETED
    prompt_output: str

class MessageMLTaskPromptResponseFailed(BaseModel):
    status: Literal[MessageMLTaskStatus.FAILED] = MessageMLTaskStatus.FAILED
    error_code: Optional[str]
    error_message: Optional[str]

class MessageMLTaskPromptResponse(MessageMLTaskPromptBase):
    response: Union[
        MessageMLTaskPromptResponseCompleted, MessageMLTaskPromptResponseFailed] = Field(..., discriminator="status")

