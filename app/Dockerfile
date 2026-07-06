FROM python:3.12-alpine

ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apk add --no-cache gcc musl-dev python3-dev libffi-dev

COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install --no-cache-dir -r /app/requirements.txt
EXPOSE 8080
COPY ./ /app

#CMD ["python", "main.py"]
#CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
# For debug:
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080", "--reload", "--log-level", "info"]