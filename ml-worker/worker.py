import os
import pika
import json
import torch
from transformers import pipeline, BitsAndBytesConfig


MODEL_ID = os.getenv("MODEL_ID", "HuggingFaceTB/SmolLM2-1.7B-Instruct")
RABBIT_URL = os.getenv("RABBITMQ_URL")

print(f"--- Loading Model: {MODEL_ID} ---")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# Pipeline init
generator = pipeline(
    "text-generation",
    model=MODEL_ID,
    model_kwargs={"quantization_config": bnb_config},
    device_map="auto"
)


def process_task(ch, method, properties, body):
    try:
        data = json.loads(body)
        user_games = data.get("prompt_input, Half Life 2, Doom 3")

        # Prompt
        messages = [
            {"role": "user", "content": f"I like these video games: {user_games}. Recommend me 1 new game according to my preferences. In your answer type only the game's title, do not type anything else."}
        ]

        output = generator(messages, max_new_tokens=100, do_sample=True, temperature=0.7)
        response_text = output[0]['generated_text'][-1]['content']

        print(f"Response: {response_text}")

        ch.basic_ack(delivery_tag=method.delivery_tag, requeue=False)

    except Exception as e:
        print(f"Error processing task: {e}")
        ch.basic_ack(delivery_tag=method.delivery_tag, requeue=False)

#-- SELF TEST
def run_test():
    print("--- Running initial self-test ---")
    test_games = "Half Life 2, Doom 3, Quake 3 Arena"
    test_messages = [
        {"role": "user",
         "content": f"I like these video games: {test_games}. Recommend me 1 new game according to my preferences. In your answer type only the game's title, do not type anything else."}
    ]
    try:
        test_output = generator(test_messages, max_new_tokens=100, do_sample=True, temperature=0.7)
        response_text = test_output[0]['generated_text'][-1]['content']
        print(f"Self test response: {response_text}")
        print(f"--- Self test completed successfully ---")
    except Exception as e:
        print(f"Self test failed: {e}")

run_test()



# RabbitMQ Configuration
connection = pika.BlockingConnection(pika.URLParameters(RABBIT_URL))
channel = connection.channel()
channel.queue_declare(queue="ml_inference", durable=True)

channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue="ml_inference", on_message_callback=process_task)

print("--- Worker is ready and waiting for tasks ---")
channel.start_consuming()

