# RAG Deployment and Infrastructure Technical Specifications

## Overview
- **Purpose**: Production-ready deployment patterns for RAG system with Docker, Kubernetes, and CI/CD integration
- **Version**: Docker Compose 2.27+, Kubernetes 1.28+
- **Last Updated**: 2025-01-12

## Key Concepts

### Deployment Architecture Layers
- **Containerization**: Multi-stage Docker builds with optimization and security
- **Orchestration**: Kubernetes deployment with auto-scaling and health checks  
- **Service Mesh**: Inter-service communication and traffic management
- **CI/CD Pipeline**: Automated testing, building, and deployment
- **Infrastructure as Code**: Terraform-based provisioning and management

### Deployment Strategies
- **Blue-Green Deployment**: Zero-downtime deployments with instant rollback
- **Canary Deployment**: Gradual rollout with traffic shifting
- **Rolling Updates**: Progressive container replacement
- **A/B Testing**: Traffic-based feature validation

## Implementation Patterns

### Pattern: Multi-Stage Docker Build for RAG Services
```dockerfile
# Multi-stage Dockerfile for RAG API Service
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash --uid 1000 rag-user

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt requirements-prod.txt ./
RUN pip install --no-cache-dir --user -r requirements-prod.txt

# Development stage (for local development and testing)
FROM base as development
COPY requirements-dev.txt ./
RUN pip install --no-cache-dir --user -r requirements-dev.txt

# Copy source code
COPY . .
RUN chown -R rag-user:rag-user /app

USER rag-user

# Add local pip packages to PATH
ENV PATH="/home/rag-user/.local/bin:$PATH"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage (optimized for production deployment)
FROM base as production

# Copy only necessary files
COPY --chown=rag-user:rag-user src/ ./src/
COPY --chown=rag-user:rag-user main.py alembic.ini ./
COPY --chown=rag-user:rag-user migrations/ ./migrations/

# Switch to non-root user
USER rag-user

# Add local pip packages to PATH
ENV PATH="/home/rag-user/.local/bin:$PATH"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Production command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# Multi-stage Dockerfile for Vector Database (Qdrant)
FROM qdrant/qdrant:v1.8.1 as qdrant-base

# Custom configuration stage
FROM qdrant-base as qdrant-production

# Copy custom configuration
COPY --chown=qdrant:qdrant config/qdrant-production.yaml /qdrant/config/production.yaml

# Set production environment
ENV QDRANT__SERVICE__GRPC_PORT=6334
ENV QDRANT__SERVICE__HTTP_PORT=6333
ENV QDRANT__LOG_LEVEL=INFO
ENV QDRANT__STORAGE__SNAPSHOTS_PATH=/qdrant/snapshots
ENV QDRANT__STORAGE__STORAGE_PATH=/qdrant/storage

# Health check for Qdrant
HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:6333/healthz || exit 1

EXPOSE 6333 6334

CMD ["./qdrant", "--config-path", "/qdrant/config/production.yaml"]
```

### Pattern: Docker Compose for Development and Production
```yaml
# docker-compose.yml - Base configuration
version: '3.8'

services:
  # Vector Database
  qdrant:
    image: qdrant/qdrant:v1.8.1
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_storage:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334
      - QDRANT__SERVICE__HTTP_PORT=6333
      - QDRANT__LOG_LEVEL=INFO
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/healthz"]
      interval: 10s
      timeout: 5s
      retries: 3
      start_period: 10s
    restart: unless-stopped
    networks:
      - rag-network

  # Redis for caching and task queues
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3
    restart: unless-stopped
    networks:
      - rag-network

  # PostgreSQL for metadata and user data
  postgres:
    image: postgres:15-alpine
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_DB=rag_system
      - POSTGRES_USER=rag_user
      - POSTGRES_PASSWORD_FILE=/run/secrets/postgres_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-db.sql:/docker-entrypoint-initdb.d/init-db.sql:ro
    secrets:
      - postgres_password
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U rag_user -d rag_system"]
      interval: 10s
      timeout: 5s
      retries: 3
    restart: unless-stopped
    networks:
      - rag-network

  # RAG API Service
  rag-api:
    build:
      context: .
      dockerfile: Dockerfile
      target: development
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://rag_user:${POSTGRES_PASSWORD}@postgres:5432/rag_system
      - REDIS_URL=redis://redis:6379/0
      - QDRANT_URL=http://qdrant:6333
      - VOYAGE_API_KEY_FILE=/run/secrets/voyage_api_key
      - LOG_LEVEL=INFO
    volumes:
      - .:/app
      - /app/__pycache__
    secrets:
      - voyage_api_key
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      qdrant:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    restart: unless-stopped
    networks:
      - rag-network

  # Document Processing Worker
  doc-processor:
    build:
      context: .
      dockerfile: Dockerfile
      target: development
    environment:
      - DATABASE_URL=postgresql://rag_user:${POSTGRES_PASSWORD}@postgres:5432/rag_system
      - REDIS_URL=redis://redis:6379/0
      - QDRANT_URL=http://qdrant:6333
      - VOYAGE_API_KEY_FILE=/run/secrets/voyage_api_key
      - LOG_LEVEL=INFO
      - WORKER_CONCURRENCY=4
    volumes:
      - .:/app
      - /app/__pycache__
    secrets:
      - voyage_api_key
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      qdrant:
        condition: service_healthy
    command: ["python", "-m", "celery", "worker", "-A", "src.workers.tasks", "--loglevel=info", "--concurrency=4"]
    restart: unless-stopped
    networks:
      - rag-network
    deploy:
      replicas: 2

  # Monitoring - Prometheus
  prometheus:
    image: prom/prometheus:v2.45.0
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./monitoring/rules/:/etc/prometheus/rules/:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=30d'
      - '--web.enable-lifecycle'
      - '--enable-feature=exemplar-storage'
    restart: unless-stopped
    networks:
      - rag-network

  # Monitoring - Grafana
  grafana:
    image: grafana/grafana:10.0.0
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD_FILE=/run/secrets/grafana_admin_password
      - GF_INSTALL_PLUGINS=redis-datasource,postgres-datasource
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning:ro
      - ./monitoring/grafana/dashboards:/etc/grafana/dashboards:ro
    secrets:
      - grafana_admin_password
    restart: unless-stopped
    networks:
      - rag-network

  # Load Balancer - NGINX
  nginx:
    image: nginx:1.25-alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
      - nginx_logs:/var/log/nginx
    depends_on:
      - rag-api
    restart: unless-stopped
    networks:
      - rag-network

# Network configuration
networks:
  rag-network:
    driver: bridge
    name: rag-system-network

# Volume configuration
volumes:
  qdrant_storage:
    driver: local
  redis_data:
    driver: local
  postgres_data:
    driver: local
  prometheus_data:
    driver: local
  grafana_data:
    driver: local
  nginx_logs:
    driver: local

# Secrets configuration
secrets:
  postgres_password:
    file: ./secrets/postgres_password.txt
  voyage_api_key:
    file: ./secrets/voyage_api_key.txt
  grafana_admin_password:
    file: ./secrets/grafana_admin_password.txt

---
# docker-compose.override.yml - Development overrides
version: '3.8'

services:
  rag-api:
    build:
      target: development
    volumes:
      - .:/app
      - /app/__pycache__
    environment:
      - DEBUG=true
      - LOG_LEVEL=DEBUG
    command: ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

  doc-processor:
    build:
      target: development
    volumes:
      - .:/app
      - /app/__pycache__
    environment:
      - DEBUG=true
      - LOG_LEVEL=DEBUG

---
# docker-compose.prod.yml - Production overrides
version: '3.8'

services:
  rag-api:
    build:
      target: production
    environment:
      - DEBUG=false
      - LOG_LEVEL=INFO
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '0.5'
          memory: 1G
        reservations:
          cpus: '0.25'
          memory: 512M
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3

  doc-processor:
    build:
      target: production
    environment:
      - DEBUG=false
      - LOG_LEVEL=INFO
    deploy:
      replicas: 4
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 1G

  qdrant:
    volumes:
      - /mnt/qdrant-data:/qdrant/storage
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G

  postgres:
    volumes:
      - /mnt/postgres-data:/var/lib/postgresql/data
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 1G
```

### Pattern: Kubernetes Deployment Manifests
```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: rag-system
  labels:
    name: rag-system
    environment: production

---
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: rag-config
  namespace: rag-system
data:
  LOG_LEVEL: "INFO"
  REDIS_URL: "redis://redis-service:6379/0"
  QDRANT_URL: "http://qdrant-service:6333"
  POSTGRES_HOST: "postgres-service"
  POSTGRES_DB: "rag_system"
  POSTGRES_USER: "rag_user"
  WORKER_CONCURRENCY: "4"

---
# secrets.yaml (apply with kubectl create secret)
apiVersion: v1
kind: Secret
metadata:
  name: rag-secrets
  namespace: rag-system
type: Opaque
data:
  postgres-password: <base64-encoded-password>
  voyage-api-key: <base64-encoded-api-key>
  grafana-admin-password: <base64-encoded-password>

---
# qdrant-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qdrant
  namespace: rag-system
  labels:
    app: qdrant
    component: vector-database
spec:
  replicas: 1
  selector:
    matchLabels:
      app: qdrant
  template:
    metadata:
      labels:
        app: qdrant
        component: vector-database
    spec:
      containers:
      - name: qdrant
        image: qdrant/qdrant:v1.8.1
        ports:
        - containerPort: 6333
          name: http
        - containerPort: 6334
          name: grpc
        env:
        - name: QDRANT__SERVICE__HTTP_PORT
          value: "6333"
        - name: QDRANT__SERVICE__GRPC_PORT
          value: "6334"
        - name: QDRANT__LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /healthz
            port: 6333
          initialDelaySeconds: 10
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /readyz
            port: 6333
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
        volumeMounts:
        - name: qdrant-storage
          mountPath: /qdrant/storage
      volumes:
      - name: qdrant-storage
        persistentVolumeClaim:
          claimName: qdrant-pvc

---
# qdrant-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: qdrant-service
  namespace: rag-system
  labels:
    app: qdrant
spec:
  selector:
    app: qdrant
  ports:
  - port: 6333
    targetPort: 6333
    protocol: TCP
    name: http
  - port: 6334
    targetPort: 6334
    protocol: TCP
    name: grpc
  type: ClusterIP

---
# qdrant-pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: qdrant-pvc
  namespace: rag-system
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: fast-ssd
  resources:
    requests:
      storage: 100Gi

---
# rag-api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api
  namespace: rag-system
  labels:
    app: rag-api
    component: api-server
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: rag-api
  template:
    metadata:
      labels:
        app: rag-api
        component: api-server
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/path: "/metrics"
        prometheus.io/port: "8000"
    spec:
      containers:
      - name: rag-api
        image: contexter/rag-api:latest
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: DATABASE_URL
          value: "postgresql://$(POSTGRES_USER):$(POSTGRES_PASSWORD)@$(POSTGRES_HOST):5432/$(POSTGRES_DB)"
        - name: POSTGRES_USER
          valueFrom:
            configMapKeyRef:
              name: rag-config
              key: POSTGRES_USER
        - name: POSTGRES_HOST
          valueFrom:
            configMapKeyRef:
              name: rag-config
              key: POSTGRES_HOST
        - name: POSTGRES_DB
          valueFrom:
            configMapKeyRef:
              name: rag-config
              key: POSTGRES_DB
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: postgres-password
        - name: VOYAGE_API_KEY
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: voyage-api-key
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: rag-config
              key: REDIS_URL
        - name: QDRANT_URL
          valueFrom:
            configMapKeyRef:
              name: rag-config
              key: QDRANT_URL
        - name: LOG_LEVEL
          valueFrom:
            configMapKeyRef:
              name: rag-config
              key: LOG_LEVEL
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
        startupProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 12

---
# rag-api-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: rag-api-service
  namespace: rag-system
  labels:
    app: rag-api
spec:
  selector:
    app: rag-api
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
  type: ClusterIP

---
# rag-api-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-api-hpa
  namespace: rag-system
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
      - type: Pods
        value: 2
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60

---
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rag-ingress
  namespace: rag-system
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
spec:
  tls:
  - hosts:
    - api.rag-system.example.com
    secretName: rag-api-tls
  rules:
  - host: api.rag-system.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: rag-api-service
            port:
              number: 80
```

## Common Gotchas

### Gotcha: Container Resource Limits and Requests
- **Problem**: Incorrect resource limits cause OOMKilled containers or resource starvation
- **Solution**: Set appropriate limits based on actual usage patterns with monitoring
- **Example**:
```yaml
# BAD: No resource limits or unrealistic limits
containers:
- name: rag-api
  image: contexter/rag-api:latest
  # No resources specified - can consume unlimited resources

# GOOD: Proper resource management based on monitoring
containers:
- name: rag-api
  image: contexter/rag-api:latest
  resources:
    requests:  # Guaranteed resources
      memory: "512Mi"
      cpu: "250m"
    limits:    # Maximum allowed resources
      memory: "1Gi"
      cpu: "500m"
  # Resource monitoring and alerting
  env:
  - name: MEMORY_REQUEST_BYTES
    valueFrom:
      resourceFieldRef:
        resource: requests.memory
  - name: MEMORY_LIMIT_BYTES
    valueFrom:
      resourceFieldRef:
        resource: limits.memory
```

### Gotcha: Health Check Configuration
- **Problem**: Improper health checks cause cascading failures or slow deployments
- **Solution**: Implement comprehensive health checks with appropriate timeouts
- **Example**:
```yaml
# BAD: No health checks or inappropriate configuration
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  # No configuration - uses defaults which may be inappropriate

# GOOD: Comprehensive health check configuration
livenessProbe:
  httpGet:
    path: /health/live
    port: 8000
    httpHeaders:
    - name: Host
      value: localhost
  initialDelaySeconds: 30  # Allow time for startup
  periodSeconds: 10        # Check every 10 seconds
  timeoutSeconds: 5        # Allow 5 seconds for response
  failureThreshold: 3      # Fail after 3 consecutive failures
  successThreshold: 1      # Succeed after 1 success

readinessProbe:
  httpGet:
    path: /health/ready
    port: 8000
  initialDelaySeconds: 5   # Quick initial check
  periodSeconds: 5         # Frequent checks during startup
  timeoutSeconds: 3        # Quick timeout
  failureThreshold: 2      # Sensitive to failures
  successThreshold: 1      # Quick recovery

startupProbe:            # Separate startup probe for slow-starting containers
  httpGet:
    path: /health/startup
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 12     # Allow up to 60 seconds for startup (12 * 5)
```

### Gotcha: Secret Management and Security
- **Problem**: Secrets stored in plain text or mounted insecurely
- **Solution**: Use proper secret management with encryption and rotation
- **Example**:
```yaml
# BAD: Secrets in plain text
env:
- name: DATABASE_PASSWORD
  value: "plaintext_password"  # Never do this!

# GOOD: Proper secret management
env:
- name: DATABASE_PASSWORD
  valueFrom:
    secretKeyRef:
      name: rag-secrets
      key: postgres-password

# BETTER: Use external secret management
env:
- name: DATABASE_PASSWORD
  valueFrom:
    secretKeyRef:
      name: vault-secrets
      key: database-password

# Secret creation with proper security
apiVersion: v1
kind: Secret
metadata:
  name: rag-secrets
  namespace: rag-system
  annotations:
    replicator.v1.mittwald.de/replicate-to: "*"  # Replicate across namespaces if needed
type: Opaque
data:
  postgres-password: <base64-encoded-value>
  voyage-api-key: <base64-encoded-value>
stringData:  # For non-base64 values (will be automatically encoded)
  config.yaml: |
    database:
      password: ${DATABASE_PASSWORD}
    external_services:
      voyage_api:
        api_key: ${VOYAGE_API_KEY}
```

### Gotcha: Persistent Volume Management
- **Problem**: Data loss due to incorrect volume configuration or lack of backup
- **Solution**: Implement proper persistent volume strategy with backup and recovery
- **Example**:
```yaml
# BAD: Using default storage without consideration
volumes:
- name: data-volume
  emptyDir: {}  # Data will be lost when pod restarts!

# GOOD: Proper persistent volume configuration
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: qdrant-pvc
  namespace: rag-system
  annotations:
    # Backup annotations
    backup.kubernetes.io/backup-schedule: "0 2 * * *"  # Daily backup at 2 AM
    backup.kubernetes.io/retention-policy: "30d"
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: fast-ssd  # Use appropriate storage class
  resources:
    requests:
      storage: 100Gi
  # Volume expansion support
  volumeMode: Filesystem

# Storage class with backup support
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd-backup
provisioner: kubernetes.io/aws-ebs
parameters:
  type: gp3
  encrypted: "true"
  # Backup configuration
  backup: "true"
  snapshot-schedule: "0 */6 * * *"  # Every 6 hours
reclaimPolicy: Retain  # Don't delete data when PVC is deleted
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer
```

## Best Practices

### Blue-Green Deployment Configuration
```yaml
# blue-green-deployment.yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: rag-api-rollout
  namespace: rag-system
spec:
  replicas: 5
  strategy:
    blueGreen:
      activeService: rag-api-service-active
      previewService: rag-api-service-preview
      autoPromotionEnabled: false
      scaleDownDelaySeconds: 30
      prePromotionAnalysis:
        templates:
        - templateName: success-rate
        args:
        - name: service-name
          value: rag-api-service-preview
      postPromotionAnalysis:
        templates:
        - templateName: success-rate
        args:
        - name: service-name
          value: rag-api-service-active
  selector:
    matchLabels:
      app: rag-api
  template:
    metadata:
      labels:
        app: rag-api
    spec:
      containers:
      - name: rag-api
        image: contexter/rag-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: 512Mi
            cpu: 250m
          limits:
            memory: 1Gi
            cpu: 500m
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
# Analysis template for deployment validation
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: success-rate
  namespace: rag-system
spec:
  args:
  - name: service-name
  metrics:
  - name: success-rate
    interval: 60s
    count: 5
    successCondition: result[0] >= 0.95
    provider:
      prometheus:
        address: http://prometheus-server:80
        query: |
          rate(http_requests_total{service="{{args.service-name}}",status=~"2.."}[5m]) /
          rate(http_requests_total{service="{{args.service-name}}"}[5m])
  - name: avg-response-time
    interval: 60s
    count: 5
    successCondition: result[0] <= 0.5
    provider:
      prometheus:
        address: http://prometheus-server:80
        query: |
          histogram_quantile(0.95,
            rate(http_request_duration_seconds_bucket{service="{{args.service-name}}"}[5m])
          )
```

### CI/CD Pipeline Configuration
```yaml
# .github/workflows/deploy.yml
name: Build and Deploy RAG System

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test_password
          POSTGRES_DB: rag_test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        cache: 'pip'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    
    - name: Run linting
      run: |
        flake8 src/ tests/
        black --check src/ tests/
        mypy src/
    
    - name: Run unit tests
      run: |
        pytest tests/unit/ --cov=src --cov-report=xml --cov-report=term
      env:
        DATABASE_URL: postgresql://postgres:test_password@localhost:5432/rag_test
        REDIS_URL: redis://localhost:6379/0
    
    - name: Run integration tests
      run: |
        pytest tests/integration/ --maxfail=3
      env:
        DATABASE_URL: postgresql://postgres:test_password@localhost:5432/rag_test
        REDIS_URL: redis://localhost:6379/0
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  security-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'

  build-and-push:
    needs: [test, security-scan]
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    
    steps:
    - name: Checkout
      uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Login to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha,prefix={{branch}}-
          type=raw,value=latest,enable={{is_default_branch}}
    
    - name: Build and push
      uses: docker/build-push-action@v5
      with:
        context: .
        platforms: linux/amd64,linux/arm64
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        target: production
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy-staging:
    if: github.ref == 'refs/heads/develop'
    needs: build-and-push
    runs-on: ubuntu-latest
    environment: staging
    
    steps:
    - name: Checkout
      uses: actions/checkout@v4
    
    - name: Configure kubectl
      uses: azure/k8s-set-context@v3
      with:
        method: kubeconfig
        kubeconfig: ${{ secrets.KUBE_CONFIG }}
        context: staging-cluster
    
    - name: Deploy to staging
      run: |
        # Update image tag in staging manifests
        sed -i "s|image: contexter/rag-api:.*|image: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:develop-${{ github.sha }}|g" k8s/staging/rag-api-deployment.yaml
        
        # Apply manifests
        kubectl apply -k k8s/staging/
        
        # Wait for rollout
        kubectl rollout status deployment/rag-api -n rag-system-staging --timeout=300s
    
    - name: Run smoke tests
      run: |
        # Wait for service to be ready
        sleep 30
        
        # Run smoke tests against staging
        pytest tests/smoke/ --base-url=https://staging-api.rag-system.example.com

  deploy-production:
    if: github.ref == 'refs/heads/main'
    needs: build-and-push
    runs-on: ubuntu-latest
    environment: production
    
    steps:
    - name: Checkout
      uses: actions/checkout@v4
    
    - name: Configure kubectl
      uses: azure/k8s-set-context@v3
      with:
        method: kubeconfig
        kubeconfig: ${{ secrets.KUBE_CONFIG }}
        context: production-cluster
    
    - name: Deploy with Argo Rollouts
      run: |
        # Update image tag in production manifests
        sed -i "s|image: contexter/rag-api:.*|image: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:main-${{ github.sha }}|g" k8s/production/rag-api-rollout.yaml
        
        # Apply rollout
        kubectl apply -f k8s/production/rag-api-rollout.yaml
        
        # Monitor rollout
        kubectl argo rollouts get rollout rag-api-rollout -n rag-system --watch
```

### Infrastructure as Code (Terraform)
```hcl
# terraform/main.tf
terraform {
  required_version = ">= 1.5"
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.20"
    }
  }
}

# EKS Cluster
resource "aws_eks_cluster" "rag_cluster" {
  name     = var.cluster_name
  role_arn = aws_iam_role.cluster_role.arn
  version  = "1.28"

  vpc_config {
    subnet_ids              = var.subnet_ids
    endpoint_private_access = true
    endpoint_public_access  = true
    public_access_cidrs    = var.allowed_cidrs
  }

  encryption_config {
    provider {
      key_arn = aws_kms_key.cluster_encryption.arn
    }
    resources = ["secrets"]
  }

  enabled_cluster_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]

  depends_on = [
    aws_iam_role_policy_attachment.cluster_policy,
    aws_iam_role_policy_attachment.vpc_resource_controller,
    aws_cloudwatch_log_group.cluster_logs,
  ]

  tags = {
    Environment = var.environment
    Application = "rag-system"
    ManagedBy   = "terraform"
  }
}

# Node Groups
resource "aws_eks_node_group" "rag_nodes" {
  cluster_name    = aws_eks_cluster.rag_cluster.name
  node_group_name = "rag-nodes"
  node_role_arn   = aws_iam_role.node_role.arn
  subnet_ids      = var.private_subnet_ids

  ami_type       = "AL2_x86_64"
  capacity_type  = "ON_DEMAND"
  disk_size      = 100
  instance_types = ["m5.large", "m5.xlarge"]

  scaling_config {
    desired_size = 3
    max_size     = 10
    min_size     = 2
  }

  update_config {
    max_unavailable = 1
  }

  # Launch template for custom configuration
  launch_template {
    id      = aws_launch_template.node_template.id
    version = aws_launch_template.node_template.latest_version
  }

  depends_on = [
    aws_iam_role_policy_attachment.node_policy,
    aws_iam_role_policy_attachment.cni_policy,
    aws_iam_role_policy_attachment.registry_policy,
  ]

  tags = {
    Environment = var.environment
    Application = "rag-system"
    ManagedBy   = "terraform"
  }
}

# Launch template for optimized node configuration
resource "aws_launch_template" "node_template" {
  name_prefix   = "rag-nodes-"
  image_id      = data.aws_ssm_parameter.eks_ami_release_version.value
  instance_type = "m5.large"

  vpc_security_group_ids = [aws_security_group.node_group.id]

  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size           = 100
      volume_type           = "gp3"
      iops                  = 3000
      throughput            = 125
      encrypted             = true
      delete_on_termination = true
    }
  }

  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    cluster_name     = aws_eks_cluster.rag_cluster.name
    cluster_endpoint = aws_eks_cluster.rag_cluster.endpoint
    cluster_ca       = aws_eks_cluster.rag_cluster.certificate_authority[0].data
  }))

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name        = "rag-node"
      Environment = var.environment
      Application = "rag-system"
    }
  }

  lifecycle {
    create_before_destroy = true
  }
}

# Helm releases for core services
resource "helm_release" "ingress_nginx" {
  name       = "ingress-nginx"
  repository = "https://kubernetes.github.io/ingress-nginx"
  chart      = "ingress-nginx"
  namespace  = "ingress-nginx"
  version    = "4.7.1"

  create_namespace = true

  values = [
    templatefile("${path.module}/helm-values/ingress-nginx.yaml", {
      load_balancer_type = var.load_balancer_type
    })
  ]

  depends_on = [aws_eks_cluster.rag_cluster]
}

resource "helm_release" "cert_manager" {
  name       = "cert-manager"
  repository = "https://charts.jetstack.io"
  chart      = "cert-manager"
  namespace  = "cert-manager"
  version    = "v1.13.1"

  create_namespace = true

  set {
    name  = "installCRDs"
    value = "true"
  }

  depends_on = [aws_eks_cluster.rag_cluster]
}

resource "helm_release" "prometheus_stack" {
  name       = "prometheus-stack"
  repository = "https://prometheus-community.github.io/helm-charts"
  chart      = "kube-prometheus-stack"
  namespace  = "monitoring"
  version    = "51.2.0"

  create_namespace = true

  values = [
    file("${path.module}/helm-values/prometheus-stack.yaml")
  ]

  depends_on = [aws_eks_cluster.rag_cluster]
}
```

## Integration Points

### Container Registry Integration
- **GitHub Container Registry**: Automated image building and pushing
- **Amazon ECR**: Private registry with vulnerability scanning
- **Multi-Architecture Builds**: Support for AMD64 and ARM64

### Kubernetes Ecosystem Integration  
- **ArgoCD**: GitOps-based continuous deployment
- **Istio Service Mesh**: Traffic management and security
- **Prometheus/Grafana**: Comprehensive monitoring and alerting
- **Cert-Manager**: Automated TLS certificate management

### Cloud Provider Integration
- **AWS EKS**: Managed Kubernetes with auto-scaling
- **Google GKE**: Container-native load balancing
- **Azure AKS**: Azure-specific integrations and managed identity

## References

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Kubernetes Deployment Documentation](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)
- [ArgoCD Documentation](https://argo-cd.readthedocs.io/)
- [Helm Chart Best Practices](https://helm.sh/docs/chart_best_practices/)
- [Terraform Kubernetes Provider](https://registry.terraform.io/providers/hashicorp/kubernetes/latest/docs)

## Related Contexts

- `technical-specs/monitoring-patterns.md` - Deployment monitoring setup
- `technical-specs/testing-patterns.md` - CI/CD testing integration
- `technical-specs/qdrant-vector-database.md` - Vector database deployment
- `technical-specs/fastapi-integration.md` - API service deployment
- `integration-guides/monitoring-integration.md` - Production monitoring setup