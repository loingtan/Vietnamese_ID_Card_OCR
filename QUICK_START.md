# Quick Setup Guide

This guide will get you up and running with the Vietnamese ID Card OCR system in minutes.

## 🚀 Option 1: Quick Start (Recommended)

### For Windows Users:
```powershell
# Clone and navigate to project
git clone <repository-url>
cd VnId-Card

# Run automated setup
make first-time-setup

# Edit configuration (add your API keys)
notepad .env

# Start the web interface
make run-streamlit
```

### For Linux/macOS Users:
```bash
# Clone and navigate to project
git clone <repository-url>
cd VnId-Card

# Install dependencies
make install
make setup-config

# Edit configuration
nano .env

# Start the web interface
make run-streamlit
```

## 🐳 Option 2: Docker (Production Ready)

```bash
# Clone the repository
git clone <repository-url>
cd VnId-Card

# Copy and edit environment
cp config/.env.example .env
# Edit .env with your settings

# Start with Docker Compose
make docker-run

# Access services:
# - Web UI: http://localhost:8501
# - API: http://localhost:8000
# - Monitoring: http://localhost:9091
```

## ☸️ Option 3: Kubernetes/K3D

```bash
# For K3D local development
make deploy-k3d

# For full Kubernetes
make deploy-k8s
```

## 📋 Configuration Checklist

1. **Copy environment file**: `cp config/.env.example .env`
2. **Set API keys**: Add your Google AI API key to `.env`
3. **Download models**: Ensure model files are in `data/models/`
4. **Set paths**: Update model paths in configuration if needed

## 🔧 Available Commands

```bash
make help                 # Show all available commands
make install             # Install dependencies
make setup-config        # Setup configuration files
make run-streamlit       # Start web interface
make run-api            # Start REST API
make docker-run         # Run with Docker
make deploy-k3d         # Deploy to K3D
make start-monitoring   # Start monitoring stack
make validate-structure # Check project structure
```

## 🌐 Access Points

| Service | URL | Description |
|---------|-----|-------------|
| **Web UI** | http://localhost:8501 | Streamlit interface |
| **REST API** | http://localhost:8000 | FastAPI server |
| **API Docs** | http://localhost:8000/docs | Interactive API documentation |
| **Monitoring** | http://localhost:9091 | Prometheus metrics |

## 🔍 Troubleshooting

### Common Issues:

1. **Models not found**: Ensure model files are in `data/models/` directory
2. **API key errors**: Check your `.env` file has the correct Google AI API key
3. **Port conflicts**: Change ports in `.env` or `docker-compose.yml`
4. **Permission errors**: Run with appropriate permissions or use Docker

### Quick Fixes:

```bash
# Validate your setup
make validate-structure

# Check if services are running
docker ps

# View logs
docker-compose -f deployment/docker/docker-compose.yml logs -f

# Restart services
make docker-run
```

## 📚 Next Steps

1. **Upload an ID card image** through the web interface
2. **Test the API** using the interactive docs at `/docs`
3. **Check monitoring** if you enabled it
4. **Read the full documentation** in `docs/PROJECT_STRUCTURE.md`

## 🎯 Production Deployment

For production deployment, see:
- `docs/DEPLOYMENT_GUIDE.md` - Comprehensive deployment guide
- `docs/K3D-DEPLOYMENT.md` - K3D specific instructions
- `docs/MONITORING_README.md` - Monitoring setup

---

**🎉 You're ready to go!** The system maintains full compatibility with the original functionality while providing a much more maintainable and scalable foundation.
