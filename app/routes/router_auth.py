from typing import List, Dict
from uuid import UUID, uuid4
from typing import List, Optional, Sequence
from decimal import Decimal
import logging
from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import DECIMAL
from sqlmodel import Session

from database.config import get_settings
from database.database import get_session
from models import *

from services.auth.pwd_utils import verify_password
from services.auth.auth import create_access_token
from services.crud import crud_user


# Configure logging
logger = logging.getLogger(__name__)

auth_router = APIRouter()
settings = get_settings()



@auth_router.post(
    path='/sign_up',
    response_model=UserRead,
    status_code=status.HTTP_200_OK,
    summary='Sign_up',
    description='Register a new user'
)
def sign_up(sign_up_data: UserCreate, session: Session = Depends(get_session)):
    """
    Registers a new user
    Args:
        sign_up_data: User registration data
        session: DB session
    Returns:
        Optional[User]: Registered user (if any)
    """
    full_name = sign_up_data.full_name

    search_by_email = crud_user.get_user_by_email(sign_up_data.email, session)
    if not search_by_email:
        new_user = crud_user.create_user(sign_up_data, session)

        try:
            session.commit()
            session.refresh(new_user)
        except Exception as e:
            session.rollback()
            raise
    else:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='User with that email already exists')

    return new_user


@auth_router.post(
    path='/sign_in',
    response_model=Token,
    status_code=status.HTTP_200_OK,
    summary='Sign_in',
    description='Log in a user and get access token'
)

def sign_in(sign_in_form_data: OAuth2PasswordRequestForm = Depends(),
            session: Session = Depends(get_session)):
    """
    Authenticates the user
    Args:
        sign_in_form_data(OAuth2PasswordRequestForm):
            username: User's email
            password: User's password
        session: DB session
    Returns:
        Optional[User]: Authenticated user (if any)
    """
    email = sign_in_form_data.username
    plain_password = sign_in_form_data.password

    if user_db := crud_user.get_user_by_email(email, session):

        if user_db.hashed_password and verify_password(plain_password, user_db.hashed_password):
            token = create_access_token(user_id=user_db.id)
            return {
                "access_token": token,
                "token_type": "bearer"
            }

    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                        detail='Incorrect email or password',
                        headers={'WWW-Authenticate': 'Bearer'})
