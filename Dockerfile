FROM python:3.14-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install \
    --no-cache-dir \
    -r requirements.txt

COPY app ./app
COPY artifacts ./artifacts

CMD ["fastapi", "run", "app/main.py", "--host", "0.0.0.0", "--port", "8000"]