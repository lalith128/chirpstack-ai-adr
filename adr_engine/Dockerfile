FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install base dependencies (this layer will be cached)
COPY requirements.txt .
RUN pip install --no-cache-dir paho-mqtt>=1.6.1 tensorflow>=2.13.0 joblib>=1.3.0 numpy>=1.24.0 pandas>=2.0.0

# Install scikit-learn separately (so we can use cached TensorFlow layer)
RUN pip install --no-cache-dir scikit-learn>=1.3.0

# Copy Model directory
COPY ../models/ ./models/

# Copy ADR engine script
COPY adr_engine_phase1.py .

# Create data directory for CSV output
RUN mkdir -p /app/data

# Run the ADR engine
CMD ["python", "-u", "adr_engine_phase1.py"]
