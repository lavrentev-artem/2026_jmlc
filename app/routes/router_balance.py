from typing import List, Dict
from uuid import UUID, uuid4
from typing import List, Optional, Sequence
from decimal import Decimal
import logging
from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy import DECIMAL
from sqlmodel import Session

from database.config import get_settings
from database.database import get_session
from models import *

from services.auth.auth import authorize, get_auth_user
from services.crud import crud_balance


# Configure logging
logger = logging.getLogger(__name__)

balance_router = APIRouter()
settings = get_settings()


#-- Common GETs

@balance_router.get(
    path='/{user_id}/amount',
    response_model=BalanceReadCurrentAmount,
    status_code=status.HTTP_200_OK,
    summary='Balance amount',
    description="Get user's balance amount",
)
def get_balance_amount_for_user(
        user_id: UUID,
        auth_user: UserRead = Depends(get_auth_user),
        session: Session = Depends(get_session)) \
        -> Optional[BalanceReadCurrentAmount]:
    """
    Retrieves a user's by user's ID
    Args:
        user_id: User UUID
        auth_user: Authenticated user
        session: DB session
    Returns:
        Optional[BalanceReadCurrentAmount]: User (if any)
    """
    #-- AUTHORIZATION
    func_eligible_groups = (UserGroup.USER, UserGroup.ADMIN)
    if not (authorized:= authorize(auth_user=auth_user, resource_owner_id=user_id, eligible_groups=func_eligible_groups)):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    #-- FUNCTION
    response = crud_balance.get_balance_current_amount(user_id, session)
    if not response:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User's balance not found")
    return response


@balance_router.get(
    path='/{user_id}/transactions',
    response_model=Sequence[BalanceReadFullHistory],
    status_code=status.HTTP_200_OK,
    summary='Balance transactions history',
    description="Get full history of user's balance transactions"
)
def get_balance_transactions(
        user_id: UUID,
        auth_user: UserRead = Depends(get_auth_user),
        session: Session = Depends(get_session)) \
        -> Sequence[Balance]:
    """
    Retrieves full history of balance transactions
    Args:
        user_id: User UUID
        auth_user: Authenticated user
        session: DB session
    Returns:
        Sequence[BalanceReadFullHistory]: Full history of balance transactions
    """
    #-- AUTHORIZATION
    func_eligible_groups = (UserGroup.USER, UserGroup.ADMIN)
    if not (authorized:= authorize(auth_user=auth_user, resource_owner_id=user_id, eligible_groups=func_eligible_groups)):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    #-- FUNCTION

    return crud_balance.get_balance_full_history(user_id, session)

#-- Operations

@balance_router.post(
    path='/{user_id}/topup',
    response_model=BalanceReadCurrentAmount,
    status_code=status.HTTP_200_OK,
    summary='Balance amount',
    description="Get user's balance amount"
)
def balance_topup(user_id: UUID,
                  topup_in: BalanceTransactionCreateInput,
                  auth_user: UserRead = Depends(get_auth_user),
                  session: Session = Depends(get_session)) \
        -> BalanceReadCurrentAmount:
    """
    Top-ups a user's balance
    Args:
        user_id: User UUID
        topup_in: Topup input
            user_id: User UUID
            transaction_amount[Decimal]: Transaction amount
            external_id[str]: ID of the transaction in external system (e.g. payment gateway)
            description: Transaction description
        auth_user: Authenticated user
        session: DB session
    Returns:
        Optional[BalanceReadCurrentAmount]: User (if any)
    """
    #-- AUTHORIZATION
    func_eligible_groups = (UserGroup.USER, UserGroup.ADMIN)
    if not (authorized:= authorize(auth_user=auth_user, resource_owner_id=user_id, eligible_groups=func_eligible_groups)):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    #-- FUNCTION
    try:
        topup_data = topup_in.model_dump()
        topup_data['user_id'] = user_id
        topup_data['transaction_type'] = BalanceTransactionType.TOPUP
        topup_db = BalanceTransactionCreate.model_validate(topup_data)
        crud_balance.create_balance_transaction(topup_db, session)
        session.commit()

        final_balance_amount = crud_balance.get_balance_current_amount(user_id, session)
        return final_balance_amount

    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Topup failed")


