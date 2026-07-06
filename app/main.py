import sys
sys.stdout.reconfigure(line_buffering=True)
import time
from datetime import datetime
import os
from pathlib import Path
from database.config import get_settings
from database.database import init_db, get_session
from services.seed.seed import seed_data


def main():
    # Basic logging in console
    # TBD: Refactor
    print(f'App started at: {datetime.now()}')
    print(f'File: {__file__}')
    print(f'Type File: {type(__file__)}')
    print(f'Path File: {Path(__file__)}')
    print(f'Type Path File: {type(Path(__file__))}')
    print(f'Path File Resolve: {Path(__file__).resolve()}')

    # DB Init
    print('Initializing Database...')
    try:
        init_db(drop_all=get_settings().DB_FORCE_RECREATE)
        print('Database initialized successfully')
    except Exception as e:
        raise

    # DB Seeding
    print('Seeding test data...')
    session_gen = get_session()
    session = next(session_gen)
    try:
        seed_data(session)
        print('Test data seeded successfully')
    finally:
        session.close()


if __name__ == '__main__':
    main()

    while True:
        print(f"Current time : {datetime.now()}")
        time.sleep(10)

