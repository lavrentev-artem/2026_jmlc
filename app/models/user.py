import uuid
from enum import Enum
from pydantic import ConfigDict, BaseModel
from sqlmodel import SQLModel, Field, Column, String, Relationship
from typing import Optional, List, TYPE_CHECKING
from datetime import datetime, timezone
from uuid import UUID, uuid4
import re


if TYPE_CHECKING:
    from app.models.balance import Balance
    from app.models.operation import Operation


#-- Enums
class UserStatus(str, Enum):
    ACTIVE = "active"
    DEACTIVATED = "deactivated"

class UserGroup(str, Enum):
    USER = "user"
    ADMIN = "admin"

#-- Base classes
class UserBase(SQLModel):
    full_name: Optional[str] = None
    email: str = Field(..., unique=True, index=True)


#-- DB classes
class User(UserBase, table=True):
    __tablename__ = "users"
    id: Optional[uuid.UUID] = Field(default_factory=uuid4, primary_key=True, nullable=False)
    hashed_password: str = Field(...)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: UserStatus = Field(
        sa_column=Column(
            String,
            default=UserStatus.ACTIVE,
            index=True)
    )
    user_group: UserGroup = Field(
        sa_column=Column(
            String,
            default=UserGroup.USER,
            index=True)
    )
    balance_transactions: List["Balance"] = Relationship(back_populates="owner")
    operations: List["Operation"] = Relationship(back_populates="owner")

#-- Service layer classes
class UserRead(UserBase):
    id: UUID
    created_at: datetime
    status: UserStatus
    user_group: UserGroup


class UserCreate(UserBase):
    password: str


#-- Auth
class AuthUser(SQLModel):
    user_id: Optional[uuid.UUID]
    group: str

class Token(BaseModel):
    access_token: str
    token_type: str