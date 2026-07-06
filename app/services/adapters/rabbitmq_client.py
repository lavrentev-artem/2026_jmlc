import aio_pika
from aio_pika import DeliveryMode, Message
from fastapi import Request
from pydantic import BaseModel
import logging
import json
import asyncio
import uuid


logger = logging.getLogger(__name__)

class RabbitMQClient:
    def __init__(self):
        self.amqp_url = None
        self.connection = None
        self.channel = None
        self.exchange = None
        self.callback_queue = None
        self._responses: dict[str, asyncio.Future] = {}

    async def connect(self, amqp_url: str):
        """
        Establish connection to RabbitMQ and initialize required entities
        Args:
            amqp_url: RabbitMQ URL
        """
        self.amqp_url = amqp_url

        try:
            self.connection = await aio_pika.connect_robust(self.amqp_url)
            self.channel = await self.connection.channel()

            # Initialize RabbitMQ entities
            # App's Exchange
            self.exchange = await self.channel.declare_exchange(
                name="ml_exchange",
                type=aio_pika.ExchangeType.DIRECT,
                durable=True,
            )

            # Callback queue for ML-Service responses
            self.callback_queue = await self.channel.declare_queue(
                name="ml_responses",
                durable=True,
            )
            await self.callback_queue.bind(self.exchange, routing_key="ml_responses")

            asyncio.create_task(self._consume_responses())

            logger.info(f"RabbitMQ connection established, Exchange initiated successfully")

        except Exception as e:
            logger.error(f"Failed to establish connection to RabbitMQ: {e}")
            raise

    async def _consume_responses(self):
        """
        Background task to consume messages from RabbitMQ
        """
        async with self.callback_queue.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process():
                    correlation_id = str(message.correlation_id)
                    if correlation_id in self._responses:
                        future = self._responses.pop(correlation_id)
                        future.set_result(json.loads(message.body.decode()))


    async def close(self):
        if self.connection:
            await self.connection.close()

    async def publish_message(self, payload: BaseModel, routing_key: str = "predict"):
        """
        Publish message to RabbitMQ Exchange
        Args:
            payload: Message body
            routing_key: Routing key
        """
        if not self.exchange:
            logger.error("Cannot publish message to RabbitMQ: Exchange is not initialized")
            return None

        loop = asyncio.get_running_loop()
        future = loop.create_future()

        try:
            message = Message(
                correlation_id=str(payload.task_id),
                body=payload.model_dump_json().encode(),
                delivery_mode=DeliveryMode.PERSISTENT,
                reply_to=self.callback_queue.name,
            )

            self._responses[str(message.correlation_id)] = future
            await self.exchange.publish(message, routing_key=routing_key)
            logger.info(f"Message sent to {routing_key}")
            return await future

        except Exception as e:
            logger.error(f"Failed to publish message to RabbitMQ: {e}")
            raise


def get_mq_client(request: Request) -> RabbitMQClient:
    client: RabbitMQClient = request.app.state.mq_client
    return client