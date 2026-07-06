from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from contextlib import asynccontextmanager

from models.user import User
from models.balance import Balance
from models.operation import Operation

from database.config import get_settings
from database.database import init_db
from services.adapters.rabbitmq_client import RabbitMQClient
from core.exceptions import BusinessLogicException

from routes.router_observability import observability_router
from routes.router_auth import auth_router
from routes.router_user import user_router
from routes.router_balance import balance_router
from routes.router_operation import operation_router
from services.seed.seed import seed_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)
settings = get_settings()
mq_client = RabbitMQClient()

@asynccontextmanager
async def app_lifespan(app: FastAPI): #-> None:
    try:
        init_db(drop_all=get_settings().DB_FORCE_RECREATE,
                seed=get_settings().DB_SEED_TEST_DATA)

        await mq_client.connect(get_settings().RABBITMQ_URL)
        app.state.mq_client = mq_client

        yield

    except Exception as e:
        logger.error(f'App startup failed: {e}')
        raise
    finally:
        await mq_client.close()
        logger.info('App shutting down...')


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI app
    Returns:
        FastAPI app instance
    """

    app = FastAPI(
        title=settings.APP_NAME,
        description=settings.APP_DESCRIPTION,
        version=settings.APP_VERSION,
        docs_url=settings.DOCS_URL,
        redoc_url=settings.REDOC_URL,
        lifespan=app_lifespan
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )

    # Register routes
    app.include_router(observability_router, prefix="/observability")
    app.include_router(auth_router, prefix="/auth")
    app.include_router(user_router, prefix="/user")
    app.include_router(balance_router, prefix="/balance")
    app.include_router(operation_router, prefix="/operation")

    # Exception handler
    @app.exception_handler(BusinessLogicException)
    async def business_logic_exception_handler(request: Request, exception: BusinessLogicException):
        return JSONResponse(
            status_code=exception.status_code,
            content={
                "error_code": exception.__class__.__name__,
                "message": exception.message
            },
        )

    @app.exception_handler(ValueError)
    @app.exception_handler(TypeError)
    async def validation_exception_handler(request: Request, exception: Exception):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error_code": "ValidationError",
                "message": str(exception)
            },
        )

    @app.exception_handler(Exception)
    async def universal_exception_handler(request: Request, exception: Exception):
        logger.error(f'FATAL: {type(exception).__name__}: {exception}')
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error_code": "SystemError",
                "message": "Service Unavailable"
            },
        )

    return app


app = create_app()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        'api:app',
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )