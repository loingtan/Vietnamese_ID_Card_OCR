# Vietnamese ID Card OCR - K3D Deployment Guide

Hướng dẫn deployment ứng dụng Vietnamese ID Card OCR trên k3d cluster.

## Yêu cầu hệ thống

### Phần mềm cần thiết
- **Docker Desktop**: Phiên bản 20.10 trở lên
- **k3d**: Phiên bản 5.6.0 trở lên
- **kubectl**: Phiên bản 1.25 trở lên
- **PowerShell** (Windows) hoặc **Bash** (Linux/macOS)

### Yêu cầu phần cứng
- RAM: Tối thiểu 8GB (khuyến nghị 16GB)
- CPU: Tối thiểu 4 cores
- Disk: Tối thiểu 10GB trống
- Network: Kết nối internet ổn định

## Cài đặt dependencies

### 1. Cài đặt k3d

#### Windows (PowerShell)
```powershell
# Sử dụng Chocolatey
choco install k3d

# Hoặc sử dụng Scoop
scoop install k3d

# Hoặc tải xuống trực tiếp
curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | bash
```

#### Linux/macOS
```bash
# Sử dụng script cài đặt
curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | bash

# Hoặc sử dụng package manager
brew install k3d  # macOS
```

### 2. Cài đặt kubectl

#### Windows
```powershell
# Sử dụng Chocolatey
choco install kubernetes-cli

# Hoặc tải xuống trực tiếp từ Kubernetes
curl -LO "https://dl.k8s.io/release/v1.28.0/bin/windows/amd64/kubectl.exe"
```

#### Linux/macOS
```bash
# Linux
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"

# macOS
brew install kubectl
```

## Deployment nhanh

### Phương pháp 1: Sử dụng PowerShell script (Windows)

```powershell
# Chạy deployment script
.\deploy-k3d.ps1

# Hoặc với các tùy chọn
.\deploy-k3d.ps1 -Force -ClusterName "my-cluster"
```

### Phương pháp 2: Sử dụng Bash script (Linux/macOS)

```bash
# Cấp quyền thực thi
chmod +x deploy-k3d.sh

# Chạy deployment script
./deploy-k3d.sh
```

### Phương pháp 3: Sử dụng Makefile

```bash
# Deployment hoàn chỉnh
make -f Makefile.k3d all

# Hoặc từng bước
make -f Makefile.k3d check-deps
make -f Makefile.k3d create-cluster
make -f Makefile.k3d build
make -f Makefile.k3d push
make -f Makefile.k3d deploy
```

## Deployment thủ công

### 1. Tạo k3d cluster

```bash
# Tạo cluster với cấu hình từ file
k3d cluster create --config k3d-config.yaml

# Hoặc tạo cluster đơn giản
k3d cluster create vnidcard-cluster \
  --agents 2 \
  --registry-create vnidcard-registry:5000 \
  --port "8501:8501@loadbalancer" \
  --port "8080:8080@loadbalancer"
```

### 2. Build và push Docker image

```bash
# Build image
docker build -t vnidcard-app:latest .

# Tag cho local registry
docker tag vnidcard-app:latest localhost:5000/vnidcard-app:latest

# Push to registry
docker push localhost:5000/vnidcard-app:latest
```

### 3. Deploy ứng dụng

```bash
# Apply tất cả manifests
kubectl apply -f namespace-rbac.yaml
kubectl apply -f persistent-volumes.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml
kubectl apply -f autoscaling.yaml

# Chờ deployment sẵn sàng
kubectl wait --for=condition=available --timeout=300s deployment/vnidcard-app
```

## Kiểm tra deployment

### 1. Kiểm tra trạng thái

```bash
# Kiểm tra pods
kubectl get pods -o wide

# Kiểm tra services
kubectl get services

# Kiểm tra deployments
kubectl get deployments

# Xem logs
kubectl logs -l app=vnidcard-app -f
```

### 2. Truy cập ứng dụng

- **Streamlit UI**: http://localhost:8501
- **API Endpoint**: http://localhost:8080
- **Registry**: http://localhost:5000

### 3. Test kết nối

```bash
# Test Streamlit
curl -f http://localhost:8501

# Test API (nếu có health endpoint)
curl -f http://localhost:8080/health
```

## Quản lý cluster

### Scale ứng dụng

```bash
# Scale to 3 replicas
kubectl scale deployment vnidcard-app --replicas=3

# Sử dụng Makefile
make -f Makefile.k3d scale REPLICAS=3
```

### Update ứng dụng

```bash
# Build new image
docker build -t vnidcard-app:v2 .
docker tag vnidcard-app:v2 localhost:5000/vnidcard-app:v2
docker push localhost:5000/vnidcard-app:v2

# Update deployment
kubectl set image deployment/vnidcard-app vnidcard-app=k3d-vnidcard-registry:5000/vnidcard-app:v2

# Rolling restart
kubectl rollout restart deployment/vnidcard-app
```

### Monitoring

```bash
# Resource usage
kubectl top nodes
kubectl top pods

# Detailed pod info
kubectl describe pod <pod-name>

# Events
kubectl get events --sort-by=.metadata.creationTimestamp
```

## Troubleshooting

### Lỗi thường gặp

#### 1. Image pull error
```bash
# Kiểm tra registry
docker pull localhost:5000/vnidcard-app:latest

# Kiểm tra image trong cluster
kubectl describe pod <pod-name>
```

#### 2. Port conflict
```bash
# Kiểm tra ports đang sử dụng
netstat -an | findstr :8501  # Windows
lsof -i :8501               # Linux/macOS

# Stop conflicting services
docker stop $(docker ps -q)
```

#### 3. Resource issues
```bash
# Kiểm tra resource limits
kubectl describe node

# Adjust resource requests/limits in deployment.yaml
```

#### 4. Network issues
```bash
# Kiểm tra k3d network
docker network ls | grep k3d

# Restart cluster
k3d cluster stop vnidcard-cluster
k3d cluster start vnidcard-cluster
```

### Debug commands

```bash
# Get shell in pod
kubectl exec -it <pod-name> -- /bin/bash

# Port forward
kubectl port-forward <pod-name> 8501:8501

# Copy files from pod
kubectl cp <pod-name>:/app/logs ./logs

# View cluster info
kubectl cluster-info dump
```

## Cleanup

### Xóa ứng dụng

```bash
# Sử dụng script
.\deploy-k3d.ps1 -Force  # Sẽ hỏi xác nhận xóa

# Sử dụng Makefile
make -f Makefile.k3d undeploy

# Thủ công
kubectl delete -f .
```

### Xóa cluster

```bash
# Xóa cluster
k3d cluster delete vnidcard-cluster

# Xóa registry
k3d registry delete vnidcard-registry

# Cleanup Docker
docker system prune -f
```

## Cấu hình nâng cao

### 1. Persistent Storage

File `persistent-volumes.yaml` đã được cấu hình để lưu trữ:
- Model files: `/app/models` (5GB)
- Application data: `/app/data` (10GB)

### 2. Auto-scaling

File `autoscaling.yaml` cấu hình HPA:
- Min replicas: 1
- Max replicas: 5
- CPU threshold: 70%
- Memory threshold: 80%

### 3. Security

File `namespace-rbac.yaml` cấu hình:
- Dedicated namespace
- Service account
- RBAC permissions
- Secrets management

### 4. Ingress

File `ingress.yaml` cấu hình:
- Host-based routing
- Path-based routing
- SSL termination (optional)

## Tích hợp CI/CD

### GitHub Actions (ví dụ)

```yaml
name: Deploy to k3d
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup k3d
        run: |
          curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | bash
      - name: Deploy
        run: |
          ./deploy-k3d.sh
```

### Monitoring và Logging

Để thêm monitoring:

```bash
# Install Prometheus và Grafana
kubectl apply -f https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/main/bundle.yaml

# Install Loki cho logging
kubectl apply -f https://raw.githubusercontent.com/grafana/loki/main/production/kustomize/loki-simple-scalable/loki-simple-scalable.yaml
```

## Tài liệu tham khảo

- [k3d Documentation](https://k3d.io/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Docker Documentation](https://docs.docker.com/)
- [kubectl Cheat Sheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/)

---

**Lưu ý**: Đây là cấu hình cho môi trường development/testing. Đối với production, cần thêm các cấu hình về security, monitoring, backup và disaster recovery.
