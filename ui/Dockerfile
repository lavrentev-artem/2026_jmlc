FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /ui

COPY requirements.txt /ui/
RUN pip install --upgrade pip && pip install --no-cache-dir -r /ui/requirements.txt
EXPOSE 8501
COPY ./ /ui

#CMD ["streamlit", "run", "main.py", "--server.address", "0.0.0.0", "--server.runOnSave", "true"]
CMD ["streamlit", "run", "main.py", "--server.address", "0.0.0.0", "--server.runOnSave", "true", "--client.showErrorDetails", "false"]
#CMD ["streamlit", "run", "main.py", "--server.address", "0.0.0.0"]