from aio_pika import IncomingMessage, Message
import aio_pika
import logging
import json


logger = logging.getLogger(__name__)

class RabbitMQClient:
    def __init__(self):
        self.amqp_url = None
        self.connection = None
        self.channel = None
        self.queue = None

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
            # Create Exchange
            exchange = await self.channel.declare_exchange(
                name="ml_exchange",
                type=aio_pika.ExchangeType.DIRECT,
                durable=True,
            )
            #Create Queue
            queue = await self.channel.declare_queue(name="ml_tasks", durable=True)
            # Create Binding
            await queue.bind(exchange, routing_key="predict")

            self.queue = queue

            logger.info(f"RabbitMQ connection established, Exchange initiated successfully")

        except Exception as e:
            logger.error(f"Failed to establish connection to RabbitMQ: {e}")
            raise

    async def close(self):
        if self.connection:
            await self.connection.close()

    async def consume(self, callback):
        """
        Monitors queue and consumes messages from RabbitMQ
        Args:
            callback:
        """
        await self.channel.set_qos(prefetch_count=1)

        # Subscribe to a queue
        await self.queue.consume(lambda msg: self._on_message(msg, callback))
        logger.info(f"Started consuming messages from RabbitMQ...")

    async def _on_message(self, message: IncomingMessage, callback):
        """
        Message processor
        Args:
            message: Message received from RabbitMQ
            callback: Callback function
        """
        async with message.process():
            # Unpack message
            payload = json.loads(message.body.decode())
            correlation_id = message.correlation_id
            reply_to = message.reply_to

            # Call inference
            result_data = await callback(payload)

            # Send reply
            if reply_to:
                body_bytes = result_data.model_dump_json().encode()
                await self.channel.default_exchange.publish(
                    Message(
                        body=body_bytes,
                        correlation_id=correlation_id,
                    ),
                    routing_key=reply_to
                )