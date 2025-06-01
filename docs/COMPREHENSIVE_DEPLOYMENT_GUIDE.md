# Vietnamese ID Card OCR - Comprehensive Deployment Guide 🚀

Hướng dẫn triển khai hoàn chỉnh cho hệ thống Vietnamese ID Card OCR với nhiều tùy chọn deployment từ development đến production.

## 📋 Mục lục

1. [Yêu cầu hệ thống](#-yêu-cầu-hệ-thống)
2. [Tổng quan các phương pháp deployment](#-tổng-quan-các-phương-pháp-deployment)
3. [Development Deployment](#-development-deployment)
4. [Docker Deployment](#-docker-deployment)
5. [Kubernetes (K8s) Deployment](#-kubernetes-k8s-deployment)
6. [K3D Local Deployment](#-k3d-local-deployment)
7. [Production Deployment](#-production-deployment)
8. [Monitoring & Logging](#-monitoring--logging)
9. [Security & Best Practices](#-security--best-practices)
10. [Troubleshooting](#-troubleshooting)

---

## 🖥️ Yêu cầu hệ thống

### Minimum Requirements
- **OS**: Windows 10/11, macOS 10.15+, Ubuntu 18.04+
- **RAM**: 4GB (khuyến nghị 8GB+)
- **Storage**: 10GB free space
- **Network**: Kết nối internet ổn định

### Development Requirements
- **Python**: 3.9+ (khuyến nghị 3.11)
- **Node.js**: 16+ (cho một số dev tools)
- **Git**: Latest version

### Production Requirements
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **Kubernetes**: 1.25+ (nếu dùng K8s)
- **kubectl**: Compatible với K8s cluster
- **k3d**: v5.6.0+ (cho local K8s)

---

## 🎯 Tổng quan các phương pháp Deployment

| Phương pháp | Use Case | Độ phức tạp | Thời gian setup |
|-------------|----------|-------------|-----------------|
| **Local Development** | Development, testing | ⭐ | 5-10 phút |
| **Docker Compose** | Local production testing | ⭐⭐ | 10-15 phút |
| **K3D** | Local Kubernetes testing | ⭐⭐⭐ | 15-20 phút |
| **Kubernetes** | Production deployment | ⭐⭐⭐⭐ | 30-60 phút |
| **Cloud Production** | Scalable production | ⭐⭐⭐⭐⭐ | 1-2 giờ |

---

## 🛠️ Development Deployment

### Quick Start cho Developer

#### Windows (PowerShell)
```powershell
# 1. Clone repository
git clone <repository-url>
cd VnId-Card

# 2. Setup environment
.\scripts\setup\quick-start.ps1

# 3. Configure API keys
notepad .env
# Thêm GEMINI_API_KEY=your_api_key_here

# 4. Start development server
make run-streamlit
# hoặc
make run-api
```

#### Linux/macOS
```bash
# 1. Clone repository
git clone <repository-url>
cd VnId-Card

# 2. Setup environment
chmod +x scripts/setup/check-prerequisites.sh
./scripts/setup/check-prerequisites.sh

# 3. Install dependencies
make install
make setup-config

# 4. Configure API keys
nano .env
# Thêm GEMINI_API_KEY=your_api_key_here

# 5. Start development server
make run-streamlit
```

### Available Development Commands

```bash
# Development utilities
make help                    # Hiển thị tất cả commands
make install                 # Cài đặt Python dependencies
make setup-config           # Setup configuration files
make validate-structure     # Kiểm tra project structure

# Running services
make run-streamlit          # Start Streamlit web UI (port 8501)
make run-api               # Start FastAPI server (port 8080)
make run-dev               # Start cả hai services

# Testing
make test                  # Run unit tests
make test-coverage         # Run tests với coverage report
make lint                  # Code linting và formatting
```

### Development Access Points
- **Streamlit UI**: http://localhost:8501
- **FastAPI API**: http://localhost:8080
- **API Documentation**: http://localhost:8080/docs
- **Redoc Documentation**: http://localhost:8080/redoc

---

## 🐳 Docker Deployment

### Simple Docker Deployment

#### 1. Build và Run Single Container
```bash
# Build image
docker build -f deployment/docker/Dockerfile -t vnid-card-ocr .

# Run container
docker run -d \
  --name vnid-card-api \
  -p 8080:8080 \
  -p 8000:8000 \
  -e GEMINI_API_KEY=your_api_key \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  vnid-card-ocr
```

#### 2. Docker Compose (Recommended)

**Basic Setup:**
```bash
# Start all services
make docker-run
# hoặc
docker-compose -f deployment/docker/docker-compose.yml up -d

# Xem logs
docker-compose logs -f

# Stop services
make docker-stop
```

**With Monitoring:**
```bash
# Start với monitoring stack
docker-compose -f deployment/docker/docker-compose.yml --profile monitoring up -d

# Access services:
# - API: http://localhost:8080
# - Grafana: http://localhost:3000 (admin/admin)
# - Prometheus: http://localhost:9090
# - MongoDB: localhost:27017
```

### Docker Services Overview

| Service | Port | Purpose | Profile |
|---------|------|---------|---------|
| **vnid-api** | 8080, 8000 | Main application | default |
| **mongodb** | 27017 | Database | default |
| **prometheus** | 9090 | Metrics collection | monitoring |
| **grafana** | 3000 | Visualization | monitoring |
| **alertmanager** | 9093 | Alert management | monitoring |
| **loki** | 3100 | Log aggregation | monitoring |
| **fluent-bit** | - | Log collection | monitoring |
| **node-exporter** | 9100 | System metrics | monitoring |
| **cadvisor** | 8080 | Container metrics | monitoring |

### Docker Environment Variables

```env
# .env file configuration
# Application
ENVIRONMENT=production
API_HOST=0.0.0.0
API_PORT=8080
METRICS_PORT=8000
LOG_LEVEL=INFO
ENABLE_STREAMLIT=false

# AI/ML
GEMINI_API_KEY=your_gemini_api_key_here

# Database
MONGODB_URI=mongodb://mongodb:27017/vnid_card_ocr

# Monitoring
PROMETHEUS_RETENTION=15d
LOKI_RETENTION=7d

# Security
JWT_SECRET_KEY=your_jwt_secret_here
ENCRYPTION_KEY=your_encryption_key_here
```

---

## ☸️ Kubernetes (K8s) Deployment

### Prerequisites
```bash
# Install kubectl
# Windows (Chocolatey)
choco install kubernetes-cli

# Linux
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/

# macOS
brew install kubectl
```

### Quick K8s Deployment
```bash
# 1. Build và push image đến registry
docker build -t your-registry/vnid-card-ocr:latest .
docker push your-registry/vnid-card-ocr:latest

# 2. Update image trong deployment.yaml
# Sửa image: your-registry/vnid-card-ocr:latest

# 3. Deploy tới Kubernetes
kubectl apply -f deployment/k8s/namespace-rbac.yaml
kubectl apply -f deployment/k8s/persistent-volumes.yaml
kubectl apply -f deployment/k8s/deployment.yaml
kubectl apply -f deployment/k8s/service.yaml
kubectl apply -f deployment/k8s/ingress.yaml
kubectl apply -f deployment/k8s/autoscaling.yaml

# 4. Verify deployment
kubectl get pods -o wide
kubectl get services
kubectl logs -f deployment/vnidcard-app
```

### Kubernetes Manifests Overview

#### 1. Namespace & RBAC (`namespace-rbac.yaml`)
```yaml
# Creates dedicated namespace với proper RBAC
apiVersion: v1
kind: Namespace
metadata:
  name: vnidcard
---
# ServiceAccount, Role, RoleBinding
```

#### 2. Persistent Volumes (`persistent-volumes.yaml`)
```yaml
# Storage cho models và data
- Model storage: 5GB
- Application data: 10GB
- Logs: 5GB
```

#### 3. Deployment (`deployment.yaml`)
```yaml
# Main application deployment
- Replicas: 1 (có thể scale)
- Resource limits: 2Gi RAM, 1 CPU
- Health checks: liveness + readiness probes
- Volume mounts: models, data, logs
```

#### 4. Service (`service.yaml`)
```yaml
# Service exposure
- API port: 8080
- Metrics port: 8000
- Type: ClusterIP (internal)
```

#### 5. Ingress (`ingress.yaml`)
```yaml
# External access
- Host-based routing
- Path-based routing
- SSL termination support
```

#### 6. Auto-scaling (`autoscaling.yaml`)
```yaml
# Horizontal Pod Autoscaler
- Min replicas: 1
- Max replicas: 5
- CPU threshold: 70%
- Memory threshold: 80%
```

### K8s Management Commands

```bash
# Scaling
kubectl scale deployment vnidcard-app --replicas=3

# Rolling updates
kubectl set image deployment/vnidcard-app vnidcard-app=new-image:tag
kubectl rollout status deployment/vnidcard-app
kubectl rollout undo deployment/vnidcard-app

# Monitoring
kubectl top nodes
kubectl top pods
kubectl describe pod <pod-name>
kubectl logs -f deployment/vnidcard-app

# Debugging
kubectl exec -it <pod-name> -- /bin/bash
kubectl port-forward <pod-name> 8080:8080
kubectl get events --sort-by=.metadata.creationTimestamp
```

---

## 🏠 K3D Local Deployment

K3D là lightweight Kubernetes distribution hoàn hảo cho local development và testing.

### Quick K3D Setup

#### Automated Deployment
```bash
# Windows
.\scripts\dev\deploy-k3d.ps1

# Linux/macOS
chmod +x scripts/dev/deploy-k3d.sh
./scripts/dev/deploy-k3d.sh

# Hoặc sử dụng Makefile
make deploy-k3d
```

#### Manual K3D Setup
```bash
# 1. Install k3d
# Windows (Chocolatey)
choco install k3d

# Linux/macOS
curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | bash

# 2. Create cluster với config
k3d cluster create --config deployment/k3d/k3d-config.yaml

# 3. Build và push image
docker build -t vnidcard-app:latest .
docker tag vnidcard-app:latest localhost:5001/vnidcard-app:latest
docker push localhost:5001/vnidcard-app:latest

# 4. Deploy application
kubectl apply -f deployment/k8s/

# 5. Wait for deployment
kubectl wait --for=condition=available --timeout=300s deployment/vnidcard-app
```

### K3D Configuration Features

```yaml
# deployment/k3d/k3d-config.yaml highlights
- Local registry: localhost:5001
- Port forwarding: 8080, 8000
- Volume mounting: Project directory
- Load balancer: Enabled
- Nodes: 1 server + 2 agents
```

### K3D Access Points
- **API**: http://localhost:8080
- **Metrics**: http://localhost:8000  
- **Registry**: http://localhost:5001
- **K8s Dashboard**: kubectl proxy

### K3D Management
```bash
# Cluster management
k3d cluster list
k3d cluster start vnidcard-cluster
k3d cluster stop vnidcard-cluster
k3d cluster delete vnidcard-cluster

# Registry management
docker images | grep localhost:5001
k3d registry list

# Node management
kubectl get nodes
kubectl describe node <node-name>
```

---

## 🌐 Production Deployment

### Cloud Deployment Options

#### 1. Google Cloud Platform (GKE)
```bash
# 1. Setup GKE cluster
gcloud container clusters create vnidcard-cluster \
  --zone=asia-southeast1-a \
  --num-nodes=3 \
  --enable-autoscaling \
  --min-nodes=1 \
  --max-nodes=10

# 2. Configure kubectl
gcloud container clusters get-credentials vnidcard-cluster --zone=asia-southeast1-a

# 3. Deploy application
kubectl apply -f deployment/k8s/
```

#### 2. Amazon EKS
```bash
# 1. Create EKS cluster (using eksctl)
eksctl create cluster \
  --name vnidcard-cluster \
  --region ap-southeast-1 \
  --nodes 3 \
  --node-type m5.large

# 2. Deploy application
kubectl apply -f deployment/k8s/
```

#### 3. Azure AKS
```bash
# 1. Create AKS cluster
az aks create \
  --resource-group vnidcard-rg \
  --name vnidcard-cluster \
  --node-count 3 \
  --enable-addons monitoring

# 2. Get credentials
az aks get-credentials --resource-group vnidcard-rg --name vnidcard-cluster

# 3. Deploy application
kubectl apply -f deployment/k8s/
```

### Production Configuration

#### Environment Variables
```yaml
# production.env
ENVIRONMENT=production
LOG_LEVEL=INFO
DEBUG=false

# Security
JWT_SECRET_KEY=complex_random_key_here
ENCRYPTION_KEY=another_complex_key_here

# Performance
WORKERS=4
MAX_WORKERS=8
TIMEOUT=300

# Resources
MEMORY_LIMIT=2Gi
CPU_LIMIT=1000m
```

#### Resource Limits
```yaml
# production resource configuration
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "2Gi" 
    cpu: "1000m"
```

#### Security Hardening
```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 10001
  fsGroup: 10001
  capabilities:
    drop:
      - ALL
  readOnlyRootFilesystem: true
  allowPrivilegeEscalation: false
```

---

## 📊 Monitoring & Logging

### Complete Monitoring Stack

#### 1. Start Monitoring
```bash
# Start full monitoring stack
docker-compose -f deployment/docker/docker-compose.yml --profile monitoring up -d

# Windows utility
.\monitor\start-monitoring.bat

# Linux/macOS utility  
./monitor/start-monitoring.sh
```

#### 2. Monitoring Services

| Service | URL | Purpose | Login |
|---------|-----|---------|-------|
| **Grafana** | http://localhost:3000 | Dashboards & Visualization | admin/admin |
| **Prometheus** | http://localhost:9090 | Metrics Collection | - |
| **Alertmanager** | http://localhost:9093 | Alert Management | - |
| **Loki** | http://localhost:3100 | Log Aggregation | - |

#### 3. Pre-configured Dashboards

**API Monitoring Dashboard:**
- Request rate và response time
- Error rate và status codes
- Model inference metrics
- Resource utilization

**System Monitoring Dashboard:**
- CPU, Memory, Disk usage
- Network I/O metrics
- Container metrics
- Host system metrics

**Logs Dashboard:**
- Real-time log streaming
- Error log filtering
- Log level distribution
- Full-text search

#### 4. Alert Rules

```yaml
# Critical alerts
- API down (>5 minutes)
- High error rate (>10%)
- Memory usage (>90%)
- Disk space (>90%)

# Warning alerts  
- Response time slow (>2s)
- Model confidence low (<70%)
- High CPU usage (>80%)
```

### Log Management

#### Log Structure
```
logs/
├── api.log          # API request/response logs
├── error.log        # Error và exception logs  
├── metrics.log      # Performance metrics
├── model.log        # Model inference logs
└── system.log       # System và infrastructure logs
```

#### Log Rotation
```bash
# Manual log cleanup
.\monitor\cleanup-logs.bat          # Windows
./monitor/cleanup-logs.sh           # Linux/macOS

# Automated rotation (daily)
# Crontab entry: 0 2 * * * /path/to/cleanup-logs.sh
```

---

## 🔒 Security & Best Practices

### API Security

#### 1. Authentication & Authorization
```python
# JWT token-based authentication
from fastapi_users import FastAPIUsers
from fastapi_users.authentication import JWTAuthentication

# API key protection
@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    api_key = request.headers.get("X-API-Key")
    if not validate_api_key(api_key):
        return Response(status_code=401)
    return await call_next(request)
```

#### 2. Input Validation
```python
# Pydantic models cho input validation
from pydantic import BaseModel, validator

class ImageUpload(BaseModel):
    file_size: int
    file_type: str
    
    @validator('file_size')
    def validate_size(cls, v):
        if v > 10 * 1024 * 1024:  # 10MB limit
            raise ValueError('File too large')
        return v
```

#### 3. Rate Limiting
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/process")
@limiter.limit("10/minute")
async def process_image():
    pass
```

### Infrastructure Security

#### 1. Container Security
```dockerfile
# Multi-stage build
FROM python:3.11-slim as builder
# ... build stage

FROM python:3.11-slim as runtime
RUN adduser --disabled-password --gecos '' appuser
USER appuser
# ... runtime stage
```

#### 2. Network Security
```yaml
# Docker network isolation
networks:
  vnidcard-network:
    driver: bridge
    internal: false
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

#### 3. Secrets Management
```bash
# Kubernetes secrets
kubectl create secret generic vnidcard-secrets \
  --from-literal=api-key=your_api_key \
  --from-literal=jwt-secret=your_jwt_secret
```

### Security Checklist

- [ ] ✅ API keys stored in environment variables
- [ ] ✅ Không commit secrets vào Git
- [ ] ✅ HTTPS enabled trong production
- [ ] ✅ Input validation và sanitization
- [ ] ✅ Rate limiting implemented
- [ ] ✅ Container security hardening
- [ ] ✅ Network segmentation
- [ ] ✅ Regular security updates
- [ ] ✅ Audit logging enabled
- [ ] ✅ Backup và disaster recovery

---

## 🔧 Troubleshooting

### Common Issues & Solutions

#### 1. Deployment Issues

**Issue: Container fails to start**
```bash
# Check container logs
docker logs <container-name>

# Check resource usage
docker stats

# Inspect container
docker inspect <container-name>

# Solution: Usually memory/CPU limits hoặc missing dependencies
```

**Issue: Pod in CrashLoopBackOff**
```bash
# Check pod logs
kubectl logs <pod-name> --previous

# Describe pod events
kubectl describe pod <pod-name>

# Check resource constraints
kubectl top pod <pod-name>

# Solution: Check resource limits, dependencies, health checks
```

#### 2. Connectivity Issues

**Issue: Service not accessible**
```bash
# Check service endpoints
kubectl get endpoints

# Port forward for debugging
kubectl port-forward <pod-name> 8080:8080

# Check network policies
kubectl get networkpolicies

# Solution: Verify service selector, port configuration
```

**Issue: Database connection failed**
```bash
# Test database connectivity
docker exec -it <db-container> mongo --eval "db.adminCommand('ismaster')"

# Check environment variables
docker exec <app-container> env | grep MONGO

# Solution: Verify connection string, network connectivity
```

#### 3. Performance Issues

**Issue: High memory usage**
```bash
# Monitor memory usage
kubectl top pods
docker stats

# Check memory limits
kubectl describe pod <pod-name> | grep -i memory

# Solution: Increase memory limits hoặc optimize application
```

**Issue: Slow API responses**
```bash
# Check API metrics
curl http://localhost:8000/metrics | grep response_time

# Monitor resource usage
kubectl top nodes

# Solution: Scale replicas, optimize code, increase resources
```

#### 4. Model Loading Issues

**Issue: Model files not found**
```bash
# Check volume mounts
kubectl describe pod <pod-name> | grep -i volume

# Verify model files
kubectl exec <pod-name> -- ls -la /app/data/models/

# Solution: Ensure volume mounts configured correctly
```

### Debug Commands Reference

#### Docker Debugging
```bash
# Container inspection
docker ps -a
docker logs <container> --tail 50 -f
docker exec -it <container> /bin/bash
docker inspect <container>
docker stats <container>

# Network debugging
docker network ls
docker network inspect <network-name>

# Volume debugging
docker volume ls
docker volume inspect <volume-name>
```

#### Kubernetes Debugging
```bash
# Pod debugging
kubectl get pods -o wide
kubectl describe pod <pod-name>
kubectl logs <pod-name> -f --previous
kubectl exec -it <pod-name> -- /bin/bash

# Service debugging
kubectl get svc -o wide
kubectl describe svc <service-name>
kubectl get endpoints <service-name>

# Resource debugging
kubectl top nodes
kubectl top pods
kubectl describe node <node-name>

# Event debugging
kubectl get events --sort-by='.lastTimestamp'
kubectl get events --field-selector type=Warning
```

#### Application Debugging
```bash
# Health checks
curl -f http://localhost:8080/health
curl -f http://localhost:8000/metrics

# API testing
curl -X POST http://localhost:8080/api/process \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test-image.jpg"

# Log analysis
tail -f logs/api.log
grep -i error logs/*.log
```

### Performance Tuning

#### 1. Resource Optimization
```yaml
# Optimal resource configuration
resources:
  requests:
    memory: "1Gi"    # Start with this
    cpu: "500m"      # Half CPU core
  limits:
    memory: "2Gi"    # Maximum memory
    cpu: "1000m"     # One CPU core max
```

#### 2. Scaling Configuration
```yaml
# Horizontal Pod Autoscaler
spec:
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80
```

#### 3. Application Optimization
```python
# Connection pooling
from sqlalchemy.pool import QueuePool
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30
)

# Caching
from functools import lru_cache

@lru_cache(maxsize=128)
def load_model(model_path: str):
    # Model loading logic
    pass
```

---

## 📞 Support & Resources

### Getting Help

1. **Documentation**: Đọc full docs trong `docs/` folder
2. **Issues**: Create GitHub issue với detailed description
3. **Logs**: Luôn include relevant logs khi report issues
4. **Environment**: Provide environment details (OS, versions, etc.)

### Useful Links

- [K3D Documentation](https://k3d.io/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Docker Documentation](https://docs.docker.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

### Quick Reference Commands

```bash
# Project commands
make help                    # Show all available commands
make validate-structure      # Validate project structure
make clean                   # Clean temporary files

# Development
make run-dev                 # Start development servers
make test                    # Run tests
make lint                    # Code quality checks

# Docker
make docker-build           # Build Docker image
make docker-run             # Run with Docker Compose
make docker-stop            # Stop Docker services

# Kubernetes
make deploy-k3d             # Deploy to K3D
make deploy-k8s             # Deploy to Kubernetes
make undeploy-k8s           # Remove from Kubernetes

# Monitoring
make start-monitoring       # Start monitoring stack
make stop-monitoring        # Stop monitoring stack
```

---

**🎉 Chúc bạn deployment thành công!** 

Deployment guide này cung cấp tất cả thông tin cần thiết để triển khai Vietnamese ID Card OCR system từ development đến production. Hãy chọn phương pháp phù hợp với needs và environment của bạn.

For more detailed information, check out the specific deployment guides:
- `K3D-DEPLOYMENT.md` for K3D specifics
- `PROJECT_STRUCTURE.md` for project organization
- `MONITORING_README.md` for monitoring setup
