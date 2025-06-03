FROM python:3.12.10-slim-bookworm

# Set working directory
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libzbar0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libfontconfig1 \
    libgtk-3-0 \
    wget \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY setup.py .
COPY *.py .
COPY src/ ./src/
COPY config/ ./config/
COPY data/ ./data/
COPY models/ ./models/
RUN mkdir -p logs data/uploads data/outputs
ENV PYTHONPATH=/app/src \
    ENVIRONMENT=production


EXPOSE 8080 8000

# Define health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Set default command
CMD ["python", "api_app.py"]
