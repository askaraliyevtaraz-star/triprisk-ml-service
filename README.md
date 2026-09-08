# TripRisk ML Service

A production-style machine learning inference service built with FastAPI, Pydantic, scikit-learn, Docker, pytest, and GitHub Actions.

## Overview

TripRisk ML Service exposes a trained machine learning model through a typed REST API.

The project demonstrates a simple production-style ML workflow with:

* model training and serialization;
* FastAPI inference endpoints;
* Pydantic request and response validation;
* configurable model settings through environment variables;
* structured logging;
* automated tests with pytest;
* test coverage checks;
* Ruff linting and formatting;
* Docker containerization;
* CI with GitHub Actions.

## Architecture

```text
Client
  |
  v
FastAPI
  |
  v
Pydantic validation
  |
  v
ModelService
  |
  v
scikit-learn model
  |
  v
PredictionResponse
```

Application structure:

```text
app/
├── main.py
├── schemas.py
├── api/
│   └── routes.py
├── core/
│   ├── config.py
│   └── logging.py
└── ml/
    └── service.py
```

## API

The service currently exposes the following endpoints:

* `GET /` — basic service information
* `GET /health` — service and model health status
* `GET /model-info` — model version, features, and decision threshold
* `GET /ready` — readiness status
* `POST /predict` — trip-risk prediction

Interactive API documentation is available at:

```text
http://localhost:8000/docs
```

## Example Prediction Request

```json
{
  "speed_mean": 80,
  "acceleration_std": 2.5,
  "harsh_braking_count": 4,
  "trip_duration_minutes": 30
}
```

Example response:

```json
{
  "risk_probability": 0.62,
  "risk_class": 1,
  "model_version": "1.0.0"
}
```

## Local Development

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

Run the application:

```bash
fastapi dev app/main.py
```

Then open:

```text
http://localhost:8000/docs
```

## Configuration

Application settings can be configured through environment variables.

Example:

```text
TRIPRISK_MODEL_PATH=artifacts/model.joblib
TRIPRISK_RISK_THRESHOLD=0.5
TRIPRISK_LOG_LEVEL=INFO
```

A sample configuration is provided in:

```text
.env.example
```

Do not commit real secrets or local `.env` files.

## Tests

Run the full test suite:

```bash
pytest
```

Run tests with coverage:

```bash
pytest --cov=app --cov-report=term-missing
```

The CI pipeline requires at least 80% code coverage.

## Code Quality

Run Ruff linting:

```bash
ruff check .
```

Automatically fix supported lint issues:

```bash
ruff check . --fix
```

Check formatting:

```bash
ruff format --check .
```

Format the project:

```bash
ruff format .
```

## Docker

Build the production image:

```bash
docker build -t triprisk-api .
```

Run the container:

```bash
docker run -d -p 8000:8000 --name triprisk-api triprisk-api
```

Check service health:

```bash
curl http://localhost:8000/health
```

Run with a custom prediction threshold:

```bash
docker run -d \
  -p 8000:8000 \
  -e TRIPRISK_RISK_THRESHOLD=0.75 \
  --name triprisk-api \
  triprisk-api
```

For Windows PowerShell:

```powershell
docker run `
  -d `
  -p 8000:8000 `
  -e TRIPRISK_RISK_THRESHOLD=0.75 `
  --name triprisk-api `
  triprisk-api
```

## Continuous Integration

GitHub Actions automatically validates the project on pushes and pull requests.

The CI pipeline performs:

```text
Git push / Pull Request
        |
        v
Ruff lint
        |
        v
Ruff format check
        |
        v
pytest
        |
        v
coverage >= 80%
        |
        v
Docker image build
        |
        v
CI passed
```

The Docker build runs only after the code-quality and test job succeeds.

## Technology Stack

### Machine Learning

* Python
* scikit-learn
* NumPy
* joblib
- Apache Airflow
- MLflow

### API

* FastAPI
* Pydantic
* Pydantic Settings
* Uvicorn

### Engineering

* pytest
* pytest-cov
* Ruff
* structured logging
* environment-based configuration

### Infrastructure

* Docker
* Git
* GitHub
* GitHub Actions


## Project Structure

```text
triprisk-ml-service/
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── logging.py
│   │
│   └── ml/
│       ├── __init__.py
│       └── service.py
│
├── artifacts/
│   └── model.joblib
│
├── scripts/
│   └── train_model.py
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_health.py
│   └── test_predict.py
│
├── .dockerignore
├── .env.example
├── .gitignore
├── Dockerfile
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Current Goals

This project is being developed as a practical ML engineering portfolio project.

Planned additions include:

* experiment tracking with MLflow;
* model registry;
* Airflow orchestration;
* PySpark preprocessing;
* cloud deployment;
* monitoring and observability.

## ML Training Pipeline

Airflow orchestrates:

prepare_data
→ parallel model training
→ model selection
→ MLflow Model Registry
→ champion model promotion