from typing import List, Dict
from uuid import UUID, uuid4
from typing import List, Optional, Sequence
import logging
from fastapi import APIRouter, HTTPException, status, Depends
from sqlmodel import Session

import models
from database.config import get_settings
from  database.database import get_session
from models import *

from services.crud import crud_user
from services.auth.auth import get_auth_user, authorize

# Configure logging
logger = logging.getLogger(__name__)

user_router = APIRouter()
settings = get_settings()


#-- Specific GETs

@user_router.get(
    path='/get_by_email',
    response_model=UserRead,
    status_code=status.HTTP_200_OK,
    summary='User',
    description='Get user info'
)
def get_user_by_email(email: str,
                      auth_user: UserRead = Depends(get_auth_user),
                      session: Session = Depends(get_session)) -> Optional[User]:
    """
    Retrieves a user by its email
    Args:
        email: User email address
        session: DB session
        auth_user: Authenticated user
    Returns:
        Optional[User]: User (if any)
    """
    user_db = crud_user.get_user_by_email(email, session)
    if not user_db:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='User not found')
    user_id = user_db.id

    #-- AUTHORIZATION
    func_eligible_groups = (UserGroup.ADMIN,)
    if not (authorized:= authorize(auth_user=auth_user, resource_owner_id=user_id, eligible_groups=func_eligible_groups)):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    #-- FUNCTION
    return user_db

@user_router.get(
    path='/me',
    response_model=UserRead,
    status_code=status.HTTP_200_OK,
    summary='Get current user',
    description='Get user info by access token'
)
def get_user_by_token(auth_user: UserRead = Depends(get_auth_user)):
    """
    Retrieves a user by an access token
    Args:
        auth_user Authenticated user
    Returns:
        auth_user: Authenticated user
    """
    return auth_user



#-- Common GETs

@user_router.get(
    path='/',
    response_model=List[UserRead],
    status_code=status.HTTP_200_OK,
    summary='Users list',
    description='Get users'
)
def get_users(auth_user: UserRead = Depends(get_auth_user),
              session: Session = Depends(get_session)) \
        -> Sequence[User]:
    """
    Args:
        auth_user: Authenticated user
        session: DB session
    Retrieve users
    Returns:
        Users list
    """
    # -- AUTHORIZATION
    func_eligible_groups = (UserGroup.ADMIN,)
    if not (authorized := authorize(auth_user=auth_user, resource_owner_id=None, eligible_groups=func_eligible_groups)):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    # -- FUNCTION
    return crud_user.get_all_users(session)


@user_router.get(
    path='/{user_id}',
    response_model=UserRead,
    status_code=status.HTTP_200_OK,
    summary='User',
    description='Get user info'
)
def get_user(user_id: UUID,
             session: Session = Depends(get_session),
             auth_user: UserRead = Depends(get_auth_user)) -> User:
    """
    Retrieve user by its UUID
    Returns:
        User info
    """
    # -- AUTHORIZATION
    func_eligible_groups = (UserGroup.USER, UserGroup.ADMIN)
    if not (authorized := authorize(auth_user=auth_user, resource_owner_id=None, eligible_groups=func_eligible_groups)):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    # -- FUNCTION
    user_db = crud_user.get_user_by_id(user_id, session)
    if not user_db:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user_db

