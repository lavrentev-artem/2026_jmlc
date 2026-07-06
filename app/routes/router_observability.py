from typing import List, Dict
import logging
from fastapi import APIRouter, HTTPException, status, Depends

from database.config import get_settings
from  database.database import get_session


# Configure logging
logger = logging.getLogger(__name__)

observability_router = APIRouter()
settings = get_settings()

@observability_router.get(
    '/healthcheck',
    response_model=Dict[str, str],
    status_code=status.HTTP_200_OK,
    summary='Healthcheck',
    description='Health check'
)
def healthcheck() -> Dict[str, str]:
    """
    Health check
    Returns:
        Dict[str, str]: App health check and basic settings
    """
    response: Dict[str, str] = {
        'status': 'ok',
        'APP_NAME': settings.APP_NAME,
        'APP_DESCRIPTION': settings.APP_DESCRIPTION,
        'APP_VERSION': settings.APP_VERSION,
        'API_VERSION': settings.API_VERSION
    }
    return response
