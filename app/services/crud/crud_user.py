from uuid import UUID, uuid4

from fastapi import Depends
from sqlmodel import Session, select
from sqlalchemy.orm import selectinload
from typing import List, Optional, Sequence


from database.database import get_session
from models import *
from services.auth.pwd_utils import hash_password

def get_all_users(session: Session) -> Sequence[User]:
    """
    Retrieves all users
    Args:
        session: DB session
    Returns:
        List[User]: List of users
    """
    try:
        statement = select(User)
        users = session.exec(statement).all()
        return users

    except Exception as e:
        raise


def get_user_by_id(user_id: UUID, session: Session) -> Optional[User]:
    """
    Retrieves a user by its ID
    Args:
        user_id: User ID
        session: DB session
    Returns:
        Optional[UserRead]: User (if any)
    """
    try:
        statement = select(User).where(User.id == user_id)
        user = session.exec(statement).one_or_none()
        return user

    except Exception as e:
        raise


def get_user_by_email(email: str, session: Session) -> Optional[User]:
    """
    Retrieves a user by its email
    Args:
        email: User email address
        session: DB session
    Returns:
        Optional[User]: User (if any)
    """
    try:
        statement =select(User).where(User.email == email)
        user = session.exec(statement).one_or_none()
        return user

    except Exception as e:
        raise


def _create_user_db(user_in: UserCreate, session: Session, user_group: UserGroup = UserGroup.USER) -> User:
    """
    Private function that creates a new user in the DB.
    Only for calling from other functions of the User service layer.
    Args:
        user_in: User data
        user_group: User group
        session: DB session
    Returns:
        Created user
    """
    try:
        user_data = user_in.model_dump()
        plain_password = user_data.pop("password")
        user_data["hashed_password"] = hash_password(plain_password)
        user_data["user_group"] = user_group
        user_data["status"] = UserStatus.ACTIVE
        user_db = User.model_validate(user_data)
        session.add(user_db)
        session.flush()
        return user_db

    except Exception as e:
        session.rollback()
        raise


def create_user(user_in: UserCreate, session: Session) -> User:
    """
    Creates a new user with group=user
    Args:
        user_in: User's data
        session: DB session
    Returns:
        Created user
    """
    return _create_user_db(user_in, user_group=UserGroup.USER, session=session)


def create_admin(user_in: UserCreate, session: Session) -> User:
    """
    Creates a new user with group=admin
    Args:
        user_in: User's data
        session: DB session
    Returns:
        Created user
    """
    return _create_user_db(user_in, user_group=UserGroup.ADMIN, session=session)