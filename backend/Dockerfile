# -----------------------------
# Base image with Python 3.11.8
# -----------------------------
FROM python:3.11.8-slim

# -----------------------------
# Set working directory inside container
# -----------------------------
WORKDIR /app

# -----------------------------
# Install system dependencies
# (for some Python packages like cffi, cryptography, etc.)
# -----------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libffi-dev \
        libssl-dev \
        libpq-dev \
        curl \
        git \
        && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Copy requirements and install
# -----------------------------
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# -----------------------------
# Copy the rest of the app code
# -----------------------------
COPY . .

# -----------------------------
# Expose the port your app uses
# -----------------------------
EXPOSE 8000

# -----------------------------
# Command to run FastAPI app
# Adjust 'main:app' to your entry point
# -----------------------------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
