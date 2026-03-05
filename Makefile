.PHONY: install run dev test test-unit test-integration health docker-build docker-up docker-down clean

install:
	pip install -r requirements.txt

run:
	uvicorn main:app --host 0.0.0.0 --port 8000

dev:
	uvicorn main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

health:
	python scripts/health_check.py

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f agent

docker-shell:
	docker compose exec app bash
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
