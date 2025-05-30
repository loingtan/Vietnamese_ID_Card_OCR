# Vietnamese ID Card OCR Docker Image

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    libfontconfig1 \
    libgtk-3-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements_windows.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy configuration
COPY config/ ./config/

# Copy source code
COPY src/ ./src/
COPY *.py ./
COPY README.md ./
COPY setup.py ./

# Copy data directory structure (without large model files)
COPY data/dictionary/ ./data/dictionary/
COPY corner_detection_model/ ./corner_detection_model/
COPY yolo_detect_text/ ./yolo_detect_text/

# Create directories for logs and data
RUN mkdir -p logs data

# Expose ports
EXPOSE 8080 8000 8501

# Environment variables
ENV PYTHONPATH=/app
ENV ENVIRONMENT=production

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Health check for Streamlit (default service)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command - can be overridden by k8s deployment
CMD ["python", "-m", "streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
