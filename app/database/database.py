from sqlmodel import SQLModel, Session, create_engine
from contextlib import contextmanager

from models.user import User
from models.balance import Balance
from models.operation import Operation
from .config import get_settings


def get_database_engine():
    """
    Creates and configures the SQLAlchemy engine

    Returns:
        Engine: Configured SQLAlchemy engine
    """

    settings = get_settings()

    engine = create_engine(
        url = settings.DATABASE_URl_psycopg,
        echo = settings.DEBUG,
        pool_size = 5,
        max_overflow = 10,
        pool_pre_ping = True,
        pool_recycle = 3600
    )
    return engine

engine = get_database_engine()

def get_session():
    with Session(engine) as session:
        try:
            yield session
        except Exception:
            session.rollback()
            raise

def init_db(drop_all: bool = False, seed: bool = False):
    """
    Initializes the database schema

    Args:
        drop_all: If true, drops all tables before creating them

    Raises:
        Exception: Any DB-related exception
    """

    try:
        engine = get_database_engine()

        if drop_all:
            SQLModel.metadata.drop_all(engine)
            print('Database successfully initialized')

        SQLModel.metadata.create_all(engine)

        if seed:
            with Session(engine) as session:
                from services.seed.seed import seed_data

                with session.no_autoflush:
                    seed_data(session)
                    session.commit()
                print('Database successfully seeded')

    except Exception as e:
        # Temporary logging via print
        print(e)
        raise
