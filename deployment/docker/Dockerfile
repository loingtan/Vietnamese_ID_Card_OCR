
FROM python:3.12.10-slim-bookworm

WORKDIR /app
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libfontconfig1 \
    libgtk-3-0 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY setup.py ./
COPY *.py ./
COPY src/ ./src/
COPY config/ ./config/
COPY data/ ./data/
RUN mkdir -p logs data/uploads data/outputs
EXPOSE 8080 8000
ENV PYTHONPATH=/app
ENV ENVIRONMENT=production
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1
CMD ["python", "api_app.py"]
