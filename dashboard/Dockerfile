# Use an official Python image as the base
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy Poetry files and install dependencies
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && poetry install --no-dev

# Copy the wine_dashboard.py file from the dashboard directory
COPY dashboard/wine_dashboard.py ./wine_dashboard.py

# Expose the Streamlit port
EXPOSE 8501

# Run the Streamlit app
CMD ["poetry", "run", "streamlit", "run", "wine_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]

