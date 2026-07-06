import decimal
from uuid import UUID, uuid4
from decimal import Decimal
from sqlmodel import Session, select
from sqlalchemy.orm import selectinload
from sqlalchemy import func, desc
from typing import List, Optional, Sequence

from models import Operation, OperationRead, OperationCreate
from models import Balance, BalanceTransactionType
import core.exceptions as exc


def get_operation(operation_id: UUID, session: Session) -> Operation:
    """
    Retrieve an operation by its UUID
    Args:
        operation_id: Operation UUID
        session: DB session
    Returns:
        operation: Operation details
    """
    try:
        statement = select(Operation).where(Operation.id == operation_id)
        operation = session.exec(statement).one_or_none()
        return operation

    except Exception as e:
        raise


def get_operation_list(user_id: UUID, session: Session) -> Sequence[Operation]:
    """
    Retrieves a list of operations by user ID
    Args:
        user_id: user UUID
        session: DB session
    Returns:
        Sequence[Operation]: List of Operations
    """
    try:
        statement = select(Operation).where(Operation.user_id == user_id)
        operation = session.exec(statement).all()
        return operation

    except Exception as e:
        raise


def create_operation(operation_in: OperationCreate, session: Session) -> Optional[Operation]:
    """
    Create a new operation
    In MVP the cost cannot be overridden by a consumer
    Args:
        operation_in: Operation data
        session: DB session
    Returns:
        operation: Operation details
    """
    operation_data = operation_in.model_dump()

    operation_db = Operation.model_validate(operation_data)
    session.add(operation_db)
    session.flush()
    return operation_db


def update_operation(operation_db: Operation, session: Session) -> Optional[Operation]:
    """
    Updates operation in DB
    Args:
        operation_db: Operation stored in DB
        session: DB session
    Returns:
        operation_db: Operation stored in DB
    """
    session.merge(operation_db)
    session.flush()
    return operation_db
