.PHONY: help install install-backend install-frontend build-frontend run dev clean

help:
	@echo "Meowl Voice commands:"
	@echo "  make install          - Install backend + frontend dependencies"
	@echo "  make run              - Single-process packaged mode on :8000"
	@echo "  make dev              - Two-process dev mode (:8000 API + :5173 frontend)"
	@echo "  make build-frontend   - Build React frontend"
	@echo "  make clean            - Remove frontend build artifacts"

install: install-backend install-frontend

install-backend:
	cd backend && pip install -r requirements.txt

install-frontend:
	cd frontend && npm install

build-frontend:
	cd frontend && npm run build

run: build-frontend
	cd backend && python main.py

dev:
	@echo "Starting API server and frontend dev server..."
	@cd backend && python main.py &
	@cd frontend && npm run dev

clean:
	rm -rf frontend/dist
