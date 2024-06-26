# Variables
DOCKER_IMAGE_NAME=kedro-document-classifier-api
DOCKER_CONTAINER_NAME=kedro-api-container

# Default target
.PHONY: all
all: build run kedro-run start-api

# Build Docker image
.PHONY: build
build:
	@docker-compose build

# Run Docker container
.PHONY: run
run:
	@docker-compose up -d

# Stop Docker container
.PHONY: stop
stop:
	@docker-compose down

# Clean up existing Docker containers
.PHONY: clean
clean:
	@docker-compose down --remove-orphans
	@docker rm -f $(DOCKER_CONTAINER_NAME) || true

# Execute kedro run inside the running container
.PHONY: kedro-run
kedro-run:
	@docker-compose exec $(DOCKER_CONTAINER_NAME) kedro run

# Start FastAPI application inside the running container
.PHONY: start-api
start-api:
	@docker-compose exec -d $(DOCKER_CONTAINER_NAME) uvicorn src.trellis_law.api:app --host 0.0.0.0 --port 8000

# Execute kedro install inside the running container (optional)
.PHONY: kedro-install
kedro-install:
	@docker-compose exec $(DOCKER_CONTAINER_NAME) kedro install

# Rebuild and run Docker container
.PHONY: rebuild
rebuild: clean build run

# Rebuild, run, execute kedro run, and start FastAPI application
.PHONY: full-build
full-build: rebuild kedro-run start-api

# Start Kedro Jupyter Notebook inside the running container
.PHONY: notebook
notebook:
	docker-compose exec -it $(DOCKER_CONTAINER_NAME) kedro jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root
