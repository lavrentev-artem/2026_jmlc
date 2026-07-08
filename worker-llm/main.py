import logging
import asyncio
import aio_pika
import json

from adapters.rabbitmq_client import RabbitMQClient
from config.config import get_settings
from mlw_services.ml_processor import process_ml_task


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


async def main():
    mq_client = RabbitMQClient()

    try:
        await mq_client.connect(get_settings().RABBITMQ_URL)
        await mq_client.channel.set_qos(prefetch_count=1)
        await mq_client.consume(callback=process_ml_task)
        logger.info("ML Worker started")

        await asyncio.Future()

    except Exception as e:
        logger.error(f"Fatal worker error: {e}")
    finally:
        await mq_client.close()
        logger.info(f"ML Worker shut down, RabbitMQ connection closed")


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info(f"ML Worker stopped by user")