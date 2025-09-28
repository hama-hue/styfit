# Use exact Python version
FROM python:3.11.8-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies
RUN pip install -r requirements.txt

# Expose port (adjust if your app uses a different one)
EXPOSE 8000

# Command to run your app (adjust if needed)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
