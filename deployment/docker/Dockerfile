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
    libfontconfig1 \
    libgtk-3-0 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY setup.py ./
COPY *.py ./

# Copy source code
COPY src/ ./src/

# Copy configuration
COPY config/ ./config/

# Copy data directory (models, dictionaries, samples)
COPY data/ ./data/

# Create additional directories for runtime
RUN mkdir -p logs data/uploads data/outputs

# Expose ports
EXPOSE 8080 8000 8501

# Environment variables
ENV PYTHONPATH=/app
ENV ENVIRONMENT=production

# Health check for Streamlit (default service)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command - run Streamlit app
CMD ["python", "-m", "streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
