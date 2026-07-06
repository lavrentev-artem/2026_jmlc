import jwt
from jwt.exceptions import InvalidTokenError, ExpiredSignatureError
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlmodel import Session
from uuid import UUID
from datetime import datetime, timedelta, timezone
from typing import Sequence, Optional

from models import *
from database.config import get_settings
from database.database import get_session
from services.crud.crud_user import get_user_by_id


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/sign_in")


def create_access_token(user_id: UUID) -> str:
    """
    Creates a JWT access token
    Args:
        user_id: User's UUID
    Returns:
        JWT access token
    """
    settings = get_settings()

    to_encode = {
        "sub": str(user_id),
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    }

    encoded_jwt = jwt.encode(
        payload=to_encode,
        key=settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM
    )

    return encoded_jwt


def get_auth_user(token: str = Depends(oauth2_scheme),
                  session: Session = Depends(get_session)) -> UserRead:
    """
    Validates the JWT access token and returns the user ID
    Args:
        token: JWT access token
        session: DB session
    Returns:
        user_id: User UUID
    """
    settings = get_settings()
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(
            jwt=token,
            key=settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )
        user_id_str: str = payload.get("sub")
        if user_id_str is None:
            raise credentials_exception
        if not (auth_user:= get_user_by_id(UUID(user_id_str), session)):
            raise credentials_exception
        return UserRead.model_validate(auth_user)

    except ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Token expired")
    except InvalidTokenError:
        raise credentials_exception


def authorize(auth_user: UserRead,
              resource_owner_id: Optional[UUID],
              eligible_groups: Sequence[UserGroup]) -> bool:
    """
    Universal authorization function. Verifies whether the authenticated user
    should have access to the demanded resources.
    Args:
        auth_user: Authenticated user according to the access token provided
        resource_owner_id: Owner of the resources access to which
                            has been demanded by the authenticated user
        eligible_groups: List of authorized user groups, declared in the consumer function
    Returns:
        True if authorized, HTTPException otherwise
    """
    #-- Check status:
    # Only active user can be authorized
    if auth_user.status is not UserStatus.ACTIVE:
        print("Authorization: user's state prohibits to perform this action.")
        return False

    #-- Check group:
    # RBAC
    if auth_user.user_group not in eligible_groups:
        print("Authorization: RBAC check - not eligible.")
        return False

    # Admin has access to any user_id
    if auth_user.user_group == UserGroup.ADMIN:
        print("Authorization: user's group is admin - access granted.")
        pass

    # User has access to himself only
    elif auth_user.user_group == UserGroup.USER:
        if not resource_owner_id == auth_user.id:
            print("Authorization: user's group is user and demanded resources belong to other owner - access not granted.")
            return False

    # Unknown group has no access
    else:
        return False

    return True