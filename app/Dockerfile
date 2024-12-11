# Use an official Python image as the base
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy Poetry files and install dependencies
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && poetry install --no-dev

# Copy application code
COPY app/ ./app/
COPY model/ ./model/

# Expose the application port
EXPOSE 8000

# Run the FastAPI app
CMD ["poetry", "run", "uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]

