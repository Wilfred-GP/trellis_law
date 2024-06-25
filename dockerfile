# Use the official Python image from the Docker Hub
FROM python:3.8-slim

# Set environment variables to prevent Python from writing pyc files to disk
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pip requirements
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_md

# Copy the rest of the project files
COPY . .

# Expose the port on which the app will run
EXPOSE 8000

# Default command to keep the container running (useful for exec commands)
CMD ["tail", "-f", "/dev/null"]
