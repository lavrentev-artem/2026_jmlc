import uuid
from fastapi import status
from fastapi.testclient import TestClient
from sqlmodel import Session

from models.user import User, UserGroup, UserRead
from services.auth.auth import get_auth_user
from api import app

def test_healthcheck(client: TestClient):
    """Проверка доступности сервиса"""
    response = client.get("/observability/healthcheck")
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["status"] == "ok"


def test_get_user_by_id_success(client: TestClient, session: Session, test_user_model: User):
    """
    Сценарий: Успешное получение свойств пользователя по его UUID.
    """
    session.add(test_user_model)
    session.commit()
    session.refresh(test_user_model)

    response = client.get(f"/user/{test_user_model.id}")

    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["id"] == str(test_user_model.id)
    assert data["email"] == test_user_model.email
    assert data["full_name"] == test_user_model.full_name
    assert "hashed_password" not in data  # Защита данных: UserRead не должен возвращать хэш пароля


def test_get_user_by_id_not_found(client: TestClient):
    """
    Сценарий: Запрос несуществующего UUID должен приводить к ошибке 404 User Not Found.
    """
    random_uuid = uuid.uuid4()

    response = client.get(f"/user/{random_uuid}")

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json()["detail"] == "User not found"

def test_get_user_by_email_as_admin(client: TestClient, session: Session, test_user_model: User):
    """Сценарий: поиск пользователя по email (от роли Admin)."""
    # Делаем пользователя админом
    test_user_model.user_group = UserGroup.ADMIN
    session.add(test_user_model)
    session.commit()

    # Переопределяем зависимость авторизации
    from services.auth.auth import get_auth_user
    from api import app
    app.dependency_overrides[get_auth_user] = lambda: UserRead.model_validate(test_user_model)

    response = client.get(f"/user/get_by_email?email={test_user_model.email}")

    assert response.status_code == status.HTTP_200_OK