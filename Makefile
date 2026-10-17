# Makefile for Wattsmart 2.0 ML Web Application
.PHONY: help install train test lint format run docker-build docker-up docker-down clean

help:
	@echo "Available commands:"
	@echo "  make install       Install dependencies and editable package"
	@echo "  make train         Run end-to-end model training, CV, tuning and evaluation"
	@echo "  make test          Run pytest suite with coverage"
	@echo "  make lint          Check code style with ruff and black"
	@echo "  make format        Auto-format code with ruff and black"
	@echo "  make run           Start Flask development server"
	@echo "  make docker-build  Build Docker image"
	@echo "  make docker-up     Start Docker container in background"
	@echo "  make docker-down   Stop Docker container"
	@echo "  make clean         Remove cache files and build artifacts"

install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt
	python -m pip install -e . --no-deps

train:
	python -m wattsmart.train

test:
	pytest -v --cov=wattsmart --cov=app tests/

lint:
	ruff check src app tests
	black --check src app tests

format:
	ruff check --fix src app tests
	black src app tests

run:
	python app/main.py

docker-build:
	docker build -t wattsmart:2.0.0 .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
