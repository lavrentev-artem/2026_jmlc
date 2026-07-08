#FROM python:3.12-alpine
FROM nvidia/cuda:12.8.0-base-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1
WORKDIR /app

#RUN apk add --no-cache gcc musl-dev python3-dev libffi-dev

# For old versions of GPU
# RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

# For new versions of GPU
RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126
COPY requirements.txt .

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r /app/requirements.txt
COPY . .

#CMD ["python3", "test_worker.py"]
CMD ["python3", "main.py"]
#CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]