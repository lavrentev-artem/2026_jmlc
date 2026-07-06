from typing import List, Dict, Union
from uuid import UUID, uuid4
from typing import List, Optional, Sequence
from decimal import Decimal
import logging
from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy import DECIMAL
from sqlmodel import Session

import core.exceptions as exc
from database.config import get_settings
from database.database import get_session
from models import *
from models.operation import OperationStatus
from services.adapters.rabbitmq_client import RabbitMQClient, get_mq_client

from services.auth.auth import get_auth_user, authorize
from services.crud.crud_balance import get_balance_current_amount, create_balance_transaction
from services.crud import crud_operation
from services.adapters.adapter_ml_service import prompt_ml_service

# Configure logging
logger = logging.getLogger(__name__)

operation_router = APIRouter()
settings = get_settings()


#-- Common GETs

@operation_router.get(
    path='/get_by_id/{operation_id}',
    response_model=OperationRead,
    status_code=status.HTTP_200_OK,
    summary='Get operation',
    description='Get operation info',
)
def get_operation(operation_id: UUID,
                  auth_user: UserRead = Depends(get_auth_user),
                  session: Session = Depends(get_session)) -> Optional[Operation]:
    """
    Retrieves an operation by its ID
    Args:
        operation_id: Operation UUID
        auth_user: Authenticated user
        session: DB session
    Returns:
        Operation: Operation object
    """
    #-- RESOURCES RETRIEVAL
    operation_db = crud_operation.get_operation(operation_id, session)
    if not operation_db:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Operation not found")
    user_id = operation_db.user_id

    #-- AUTHORIZATION
    func_eligible_groups = (UserGroup.USER, UserGroup.ADMIN)
    if not (authorized:= authorize(auth_user=auth_user, resource_owner_id=user_id, eligible_groups=func_eligible_groups)):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    #-- FUNCTION
    return operation_db


@operation_router.get(
    path='/list',
    response_model=List[OperationRead],
    status_code=status.HTTP_200_OK,
    summary='Get operations list',
    description=f'Get operations list for a user',
)
def get_operation_list(user_id: UUID,
                       auth_user: UserRead = Depends(get_auth_user),
                       session: Session = Depends(get_session)) -> Sequence[Operation]:
    """
    Retrieves a list of operations by user ID
    Args:
        user_id: user UUID
        auth_user: Authenticated user
        session: DB session
    Returns:
        Sequence[Operation]: List of Operations
    """
    #-- AUTHORIZATION
    func_eligible_groups = (UserGroup.USER, UserGroup.ADMIN)
    if not (authorized:= authorize(auth_user=auth_user, resource_owner_id=user_id, eligible_groups=func_eligible_groups)):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    #-- FUNCTION
    return crud_operation.get_operation_list(user_id, session)


#-- Operations

@operation_router.post(
    path='/prompt',
    response_model=Union[OperationRead, CreatePromptResponse],
    status_code=status.HTTP_200_OK,
    summary='Get operation',
    description='Get operation info',
)
async def create_prompt(
        operation_input: OperationCreateInput,
        auth_user: UserRead = Depends(get_auth_user),
        session: Session = Depends(get_session),
        mq_client: RabbitMQClient = Depends(get_mq_client)
    ) -> Union[Optional[Operation], CreatePromptResponse]:
    """
    Creates a new prompt
    Args:
        operation_input: Operation input data
        auth_user: Authenticated user
        session: DB session
        mq_client: RabbitMQ client
    Returns:
        Operation: Operation object
    """
    #-- AUTHORIZATION
    func_eligible_groups = (UserGroup.USER, UserGroup.ADMIN)
    if not (authorized:= authorize(auth_user=auth_user, resource_owner_id=operation_input.user_id, eligible_groups=func_eligible_groups)):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    #-- FUNCTION
    operation_data = operation_input.model_dump()

    balance_amount = get_balance_current_amount(operation_data["user_id"], session)
    operation_cost = get_operation_cost()

    if balance_amount.current_amount < operation_cost:
        raise exc.InsufficientCredits()

    # Create pending operation
    operation_data["operation_output"] = None
    operation_data['type'] = OperationType.PROMPT
    operation_data['status'] = OperationStatus.PENDING
    operation_data["cost"] = None
    operation_data = OperationCreate.model_validate(operation_data)
    operation_db = crud_operation.create_operation(operation_data, session)
    session.commit()

    try:
        # Prompt ML service
        prompt_result = await prompt_ml_service(
            task_id=operation_db.id,
            task_created_at=operation_db.created_at,
            prompt_input=operation_data.operation_input,
            mq_client=mq_client
        )
        logger.info(f"prompt_output1 = {prompt_result}")

        if prompt_result["status"] == "failed":
            raise exc.MLPromptFailed()

        # Charging balance
        balance_charge_in = BalanceTransactionCreate.model_validate(
            {
                "user_id": operation_db.user_id,
                "transaction_type": BalanceTransactionType.CHARGE,
                "transaction_amount": -operation_cost,
                "external_id": str(operation_db.id),
                "description": "Charge for prompt"
            }
        )
        balance_charge = create_balance_transaction(
            balance_charge_in,
            session)

        # Completing operation
        operation_db.operation_output = prompt_result["prompt_output"]
        operation_db.status = OperationStatus.COMPLETE
        operation_db.cost = operation_cost
        operation_db = crud_operation.update_operation(operation_db, session)
        session.commit()

        # if not operation_created:
        #     raise exc.OperationFailed()

        return operation_db

    except Exception as e:
        try:
            session.rollback()
            operation_db.status = OperationStatus.CANCELED
            operation_db.cost = None
            operation_db.operation_output = "< ERROR >"
            operation_db = crud_operation.update_operation(operation_db, session)
            session.commit()
            logger.error(f"Operation {operation_db.id} canceled due to error: {e}")
        except Exception as db_e:
            logger.error(f"CRITICAL DB FAILURE: Operation {operation_db.id} was not canceled! DB error: {db_e}")
        return CreatePromptResponse(
            status="failed",
            error_code="prompt_processing_error",
            error_message="Prompt processing error"
        )


def get_operation_cost() -> Decimal:
    """
    Calculates the cost of the operation.
    In MVP the cost is constant = 1.00
    """
    return Decimal("1.00")


