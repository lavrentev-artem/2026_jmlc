from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    # RabbitMQ Settings
    RABBITMQ_HOST: Optional[str] = None
    RABBITMQ_PORT: Optional[str] = None
    RABBITMQ_NAME: Optional[str] = None
    RABBITMQ_PASS: Optional[str] = None

    # Application configuration
    APP_NAME: Optional[str] = None
    APP_DESCRIPTION: Optional[str] = None
    APP_VERSION: Optional[str] = None
    API_VERSION: Optional[str] = None
    DOCS_URL: Optional[str] = None
    REDOC_URL: Optional[str] = None

    DEBUG: Optional[bool] = None


    @property
    def RABBITMQ_URL(self):
        return f'amqp://{self.RABBITMQ_NAME}:{self.RABBITMQ_PASS}@{self.RABBITMQ_HOST}:{self.RABBITMQ_PORT}'

    model_config = SettingsConfigDict(
        env_file ='../.env',
        env_file_encoding = 'utf-8',
        case_sensitive = True
    )

@lru_cache()
def get_settings() -> Settings:
    settings = Settings()
    return settings
