from decimal import Decimal
from enum import Enum
from pydantic import ConfigDict, BaseModel
from sqlmodel import SQLModel, Field, Column, String, Numeric, Relationship
from typing import Optional, List, TYPE_CHECKING
from datetime import datetime, timezone
from uuid import UUID, uuid4


if TYPE_CHECKING:
    from app.models import User
    from app.models import Balance


#-- Enums
class OperationType(str, Enum):
    PROMPT = "prompt"   # basic prompt to ML-service

class OperationStatus(str, Enum):
    PENDING = "pending"   # operation output processing in progress
    COMPLETE = "complete" # operation complete
    CANCELED = "canceled" # operation canceled


#-- Base classes
class OperationBase(SQLModel):
    """
    Base class for operations.
    In MVP the cost is constant = 1.00. Later it will be retrieved from catalog.
    Cost should not be overridden by consumers.
    """
    operation_input: Optional[str] = None
    operation_output: Optional[str] = None


#-- DB classes
class Operation(OperationBase, table=True):
    __tablename__ = "operations"
    id: Optional[UUID] = Field(default_factory=uuid4, primary_key=True, nullable=False)
    type: OperationType = Field(
        sa_column=Column(
            String,
            index=True)
    )
    status: OperationStatus = Field(
        sa_column=Column(
            String,
            index=True)
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    cost: Optional[Decimal] = Field(
        max_digits=12,
        decimal_places=2,
        sa_column=Column(
            Numeric(
                precision=12,
                scale=2)
        )
    )
    user_id: UUID = Field(
        foreign_key="users.id",
        index=True)
    owner: Optional["User"] = Relationship(back_populates="operations")


#-- Service layer classes
class OperationRead(OperationBase):
    id: UUID
    type: OperationType
    cost: Optional[Decimal] = Field(
        max_digits=12,
        decimal_places=2
    )
    created_at: datetime
    user_id: UUID

class OperationCreateInput(SQLModel):
    user_id: UUID
    operation_input: str

class OperationCreate(OperationCreateInput):
    type: OperationType
    status: OperationStatus
    operation_output: Optional[str]
    cost: Optional[Decimal] = Field(
        max_digits=12,
        decimal_places=2
    )

class OperationUpdate(SQLModel):
    id: UUID
    user_id: UUID
    type: OperationType
    status: OperationStatus
    cost: Optional[Decimal] = Field(
        max_digits=12,
        decimal_places=2
    )
    operation_input: str
    operation_output: Optional[str]


#-- Negative response from API router
class CreatePromptResponse(BaseModel):
    status: str
    error_code: Optional[str] = None
    error_message: Optional[str] = None