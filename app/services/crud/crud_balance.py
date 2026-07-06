from uuid import UUID, uuid4
from decimal import Decimal
from sqlmodel import Session, select
from sqlalchemy.orm import selectinload
from sqlalchemy import func, desc
from typing import List, Optional, Sequence

from models.balance import Balance, BalanceReadCurrentAmount, BalanceReadFullHistory, BalanceTransactionRead, BalanceTransactionCreate


def get_balance_current_amount(user_id: UUID, session: Session) -> BalanceReadCurrentAmount:
    """
    Calculates the current balance amount of the user
    Args:
        user_id: User ID
        session: DB session
    Returns:
        current_amount: Balance amount
    """
    try:
        statement = (
            select(func.sum(Balance.transaction_amount))
            .where(Balance.user_id == user_id)
        )
        amount = session.exec(statement).one()
        amount = amount if amount else Decimal("0.00")

        return BalanceReadCurrentAmount(current_amount=amount)

    except Exception as e:
        raise


def get_balance_full_history(user_id: UUID, session: Session) -> Sequence[Balance]:
    """
    Retrieves history of all balance transactions of the user
    Args:
        user_id: User ID
        session: DB session
    Returns:
        balance_transactions: All balance transactions
    """
    try:
        statement = (
            select(Balance)
            .where(Balance.user_id == user_id)
            .order_by(desc(Balance.created_at))
        )
        balance_history = session.exec(statement).all()
        return balance_history

    except Exception as e:
        raise


def get_balance_transaction(transaction_id: UUID, session: Session) -> Balance:
    """
    Retrieves history of all balance transactions of the user
    Args:
        transaction_id: Balance transaction ID
        session: DB session
    Returns:
        balance_transaction: balance transactions retrieved by ID
    """
    try:
        statement = (
            select(Balance)
            .where(Balance.id == transaction_id)
        )
        balance_transaction = session.exec(statement).one()
        return balance_transaction

    except Exception as e:
        raise


def create_balance_transaction(
        balance_transaction_in: BalanceTransactionCreate,
        session: Session) -> Balance:
    """
    Retrieves history of all balance transactions of the user
    Args:
        balance_transaction_in:
            transaction_amount: Amount of the transaction
            transaction_type: type of transaction (source process)
            description: Description of the transaction
        session: DB session
    Returns:
        balance_transaction_out: balance transactions retrieved by ID
    """
    try:
        balance_transaction_db = Balance.model_validate(balance_transaction_in)
        session.add(balance_transaction_db)
        session.flush()
        return balance_transaction_db

    except Exception as e:
        raise
