from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    # DB Settings
    DB_HOST: Optional[str] = None
    DB_PORT: Optional[str] = None
    DB_USER: Optional[str] = None
    DB_PASS: Optional[str] = None
    DB_NAME: Optional[str] = None

    # RabbitMQ Settings
    RABBITMQ_HOST: Optional[str] = None
    RABBITMQ_PORT: Optional[str] = None
    RABBITMQ_NAME: Optional[str] = None
    RABBITMQ_PASS: Optional[str] = None

    # Application auth settings
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int

    # Application configuration
    APP_NAME: Optional[str] = None
    APP_DESCRIPTION: Optional[str] = None
    APP_VERSION: Optional[str] = None
    DOCS_URL: Optional[str] = None
    REDOC_URL: Optional[str] = None

    DEBUG: Optional[bool] = None
    API_VERSION: Optional[str] = None
    DB_FORCE_RECREATE: Optional[bool] = None
    DB_SEED_TEST_DATA: Optional[bool] = None

    @property
    def DATABASE_URl_asyncpg(self):
        return f'postgresql+asyncpg://{self.DB_USER}:{self.DB_PASS}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}'

    @property
    def DATABASE_URl_psycopg(self):
        return f'postgresql+psycopg://{self.DB_USER}:{self.DB_PASS}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}'

    @property
    def RABBITMQ_URL(self):
        return f'amqp://{self.RABBITMQ_NAME}:{self.RABBITMQ_PASS}@{self.RABBITMQ_HOST}:{self.RABBITMQ_PORT}'

    model_config = SettingsConfigDict(
        env_file = '.env',
        env_file_encoding = 'utf-8',
        case_sensitive = True
    )

    def validate(self):
        """
        Validation of DB configuration settings
        """
        if not all([self.DB_USER, self.DB_PASS, self.DB_HOST, self.DB_NAME]):
            raise ValueError('Missing required DB configuration settings')

@lru_cache()
def get_settings() -> Settings:
    settings = Settings()
    settings.validate()
    return settings
