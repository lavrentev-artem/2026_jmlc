import pytest
from unittest.mock import AsyncMock
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, Session, create_engine
from sqlalchemy.pool import StaticPool
import uuid

from database.database import get_session
from services.adapters.rabbitmq_client import RabbitMQClient
from services.auth.auth import get_auth_user
from models.user import User, UserStatus, UserGroup, UserRead
from api import app


@pytest.fixture(name="session", scope="function")
def session_fixture():
    """Создает чистую БД в памяти для каждого теста."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    SQLModel.metadata.drop_all(engine)


@pytest.fixture(name="test_user_model")
def test_user_model_fixture():
    """Базовый объект пользователя, который мы будем использовать для авторизации и тестов."""
    return User(
        id=uuid.uuid4(),
        email="developer_mentor@mit.edu",
        full_name="MIT Professor",
        hashed_password="fake_secure_hash",
        status=UserStatus.ACTIVE,
        user_group=UserGroup.USER
    )


@pytest.fixture(name="client", scope="function")
def client_fixture(session: Session, test_user_model: User, monkeypatch):
    """Клиент с подмененными сетевыми клиентами и базовой базой данных."""
    # Отключаем RabbitMQ
    monkeypatch.setattr(RabbitMQClient, "connect", AsyncMock())
    monkeypatch.setattr(RabbitMQClient, "close", AsyncMock())

    # По умолчанию подставляем тестовую сессию БД
    app.dependency_overrides[get_session] = lambda: session

    # По умолчанию подставляем авторизованного пользователя как UserRead
    authenticated_user_dto = UserRead.model_validate(test_user_model)
    app.dependency_overrides[get_auth_user] = lambda: authenticated_user_dto

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()