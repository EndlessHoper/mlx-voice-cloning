.PHONY: help install install-backend install-frontend build-frontend run dev clean

help:
	@echo "mlx-voice-cloning commands:"
	@echo "  make install          - Install backend + frontend dependencies"
	@echo "  make run              - Build frontend, serve everything on :8000"
	@echo "  make dev              - Dev mode: API on :8000, frontend on :5173 with hot reload"
	@echo "  make clean            - Remove frontend build artifacts"

install: install-backend install-frontend

install-backend:
	cd backend && uv venv && uv pip install -r requirements.txt

install-frontend:
	cd frontend && npm install

build-frontend:
	cd frontend && npm run build

run: build-frontend
	cd backend && uv run python main.py

dev:
	cd backend && uv run python main.py &
	cd frontend && npm run dev

clean:
	rm -rf frontend/dist
