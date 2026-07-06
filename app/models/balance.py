import uuid
from decimal import Decimal
from enum import Enum
from pydantic import ConfigDict
from sqlmodel import SQLModel, Field, Column, String, Relationship
from typing import Optional, List, TYPE_CHECKING
from datetime import datetime, timezone
from uuid import UUID, uuid4


if TYPE_CHECKING:
    from app.models.user import User


#-- Enums
class BalanceTransactionType(str, Enum):
    TOPUP = "topup"     # Balance topup
    CHARGE = "charge"   # Balance charge

#-- Base classes
class BalanceBase(SQLModel):
    transaction_amount: Decimal
    external_id: Optional[str] = None
    description: Optional[str] = None


#-- DB classes
class Balance(BalanceBase, table=True):
    __tablename__ = "balance_transactions"
    id: Optional[uuid.UUID] = Field(
        default_factory=uuid4,
        primary_key=True,
        nullable=False)
    transaction_type: BalanceTransactionType = Field(
        sa_column=Column(
            String,
            index=True)
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        index=True)
    user_id: uuid.UUID = Field(
        foreign_key="users.id",
        index=True)
    owner: Optional["User"] = Relationship(back_populates="balance_transactions")


#-- Service layer classes
class BalanceReadCurrentAmount(SQLModel):
    current_amount: Decimal = Field(default=Decimal("0.00"))


class BalanceReadFullHistory(BalanceBase):
    id: uuid.UUID
    transaction_type: BalanceTransactionType
    created_at: datetime
    user_id: uuid.UUID


class BalanceTransactionRead(BalanceBase):
    id: uuid.UUID
    transaction_type: BalanceTransactionType
    created_at: datetime
    user_id: uuid.UUID


class BalanceTransactionCreate(BalanceBase):
    user_id: uuid.UUID
    transaction_type: BalanceTransactionType

class BalanceTransactionCreateInput(SQLModel):
    # user_id: uuid.UUID
    transaction_amount: Decimal
    external_id: Optional[str] = None
    description: Optional[str] = None