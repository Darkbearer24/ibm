# Speech Translation System - Deployment Guide

## Table of Contents

1. [Overview](#overview)
2. [Pre-deployment Checklist](#pre-deployment-checklist)
3. [Environment Setup](#environment-setup)
4. [Local Development Deployment](#local-development-deployment)
5. [Production Deployment](#production-deployment)
6. [Docker Deployment](#docker-deployment)
7. [Cloud Deployment](#cloud-deployment)
8. [Monitoring and Maintenance](#monitoring-and-maintenance)
9. [Security Considerations](#security-considerations)
10. [Troubleshooting](#troubleshooting)
11. [Scaling and Performance](#scaling-and-performance)

---

## Overview

This guide provides comprehensive instructions for deploying the Speech Translation System in various environments, from local development to production-scale deployments.

### Deployment Options

- **Local Development**: Single-machine setup for development and testing
- **Production Server**: Dedicated server deployment with monitoring
- **Docker Container**: Containerized deployment for consistency
- **Cloud Platform**: Scalable cloud deployment (AWS, Azure, GCP)
- **Kubernetes**: Container orchestration for high availability

### System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Interface │    │  Pipeline        │    │  Model          │
│   (Streamlit)   │───▶│  Orchestrator    │───▶│  Components     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   File Storage  │    │  Logging &       │    │  Audio          │
│   & Results     │    │  Monitoring      │    │  Processing     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

---

## Pre-deployment Checklist

### ✅ System Requirements Verification

**Hardware Requirements:**
- [ ] CPU: 4+ cores (8+ recommended for production)
- [ ] RAM: 8GB minimum (16GB+ recommended for production)
- [ ] Storage: 10GB free space (50GB+ for production)
- [ ] Network: Stable internet connection

**Software Requirements:**
- [ ] Python 3.8+ installed
- [ ] Git installed
- [ ] Virtual environment support
- [ ] Web browser (for interface access)

### ✅ Performance Validation

```bash
# Run performance benchmarks
python benchmark_performance.py

# Expected results:
# - Processing speed: >1x real-time
# - Memory usage: <2GB peak
# - Success rate: >95%
```

### ✅ Testing Validation

```bash
# Run comprehensive test suite
python run_tests.py

# Expected results:
# - Unit tests: PASS
# - Integration tests: PASS
# - Edge case tests: PASS
```

### ✅ Security Checklist

- [ ] No hardcoded secrets or API keys
- [ ] Secure file upload validation
- [ ] Input sanitization implemented
- [ ] Logging configured (no sensitive data)
- [ ] Error handling prevents information leakage

---

## Environment Setup

### Development Environment

```bash
# Clone repository
git clone <repository-url>
cd speech-translation-system

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from utils.pipeline_orchestrator import PipelineOrchestrator; print('Setup complete!')"
```

### Environment Variables

Create a `.env` file in the project root:

```bash
# Application Configuration
APP_ENV=development
DEBUG=true
LOG_LEVEL=INFO

# Directories
OUTPUT_DIR=outputs
LOG_DIR=logs
MODEL_CACHE_DIR=models
UPLOAD_DIR=uploads

# Processing Configuration
MAX_AUDIO_LENGTH=300
MAX_FILE_SIZE=100MB
DEFAULT_SAMPLE_RATE=44100

# Performance
DEVICE=cpu
MAX_CONCURRENT_REQUESTS=4
CACHE_ENABLED=true

# Security
ALLOWED_EXTENSIONS=wav,mp3,flac,m4a
MAX_UPLOAD_SIZE=104857600
SECURE_UPLOADS=true
```

### Production Environment Variables

```bash
# Production Configuration
APP_ENV=production
DEBUG=false
LOG_LEVEL=WARNING

# Security
SECRET_KEY=your-secure-secret-key
ALLOWED_HOSTS=your-domain.com,www.your-domain.com
SSL_ENABLED=true

# Database (if using)
DATABASE_URL=postgresql://user:pass@localhost/dbname

# Monitoring
MONITORING_ENABLED=true
METRICS_ENDPOINT=/metrics
HEALTH_CHECK_ENDPOINT=/health

# External Services
REDIS_URL=redis://localhost:6379
CELERY_BROKER_URL=redis://localhost:6379
```

---

## Local Development Deployment

### Quick Start

```bash
# Start the web interface
streamlit run app.py

# Access the application
# Open browser to: http://localhost:8501
```

### Development Server Configuration

```bash
# Custom port and host
streamlit run app.py --server.port 8080 --server.address 0.0.0.0

# Enable auto-reload for development
streamlit run app.py --server.runOnSave true

# Debug mode
streamlit run app.py --logger.level debug
```

### Development Workflow

1. **Code Changes**: Make your modifications
2. **Testing**: Run unit and integration tests
3. **Local Testing**: Test via web interface
4. **Performance Check**: Run benchmarks if needed
5. **Commit**: Commit changes to version control

---

## Production Deployment

### Server Setup

#### Ubuntu/Debian Server

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install python3 python3-pip python3-venv git nginx -y

# Install system audio libraries
sudo apt install libsndfile1 ffmpeg -y

# Create application user
sudo useradd -m -s /bin/bash speechapp
sudo su - speechapp
```

#### CentOS/RHEL Server

```bash
# Update system
sudo yum update -y

# Install Python and dependencies
sudo yum install python3 python3-pip git nginx -y

# Install EPEL repository for additional packages
sudo yum install epel-release -y
sudo yum install ffmpeg -y
```

### Application Deployment

```bash
# Clone application
git clone <repository-url> /opt/speech-translation
cd /opt/speech-translation

# Set ownership
sudo chown -R speechapp:speechapp /opt/speech-translation

# Switch to application user
sudo su - speechapp
cd /opt/speech-translation

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p outputs logs uploads models

# Set permissions
chmod 755 outputs logs uploads
```

### Process Management with Systemd

Create `/etc/systemd/system/speech-translation.service`:

```ini
[Unit]
Description=Speech Translation System
After=network.target

[Service]
Type=simple
User=speechapp
Group=speechapp
WorkingDirectory=/opt/speech-translation
Environment=PATH=/opt/speech-translation/.venv/bin
EnvironmentFile=/opt/speech-translation/.env
ExecStart=/opt/speech-translation/.venv/bin/streamlit run app.py --server.port 8501 --server.address 127.0.0.1
Restart=always
RestartSec=10

# Security settings
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ReadWritePaths=/opt/speech-translation/outputs /opt/speech-translation/logs /opt/speech-translation/uploads

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable speech-translation
sudo systemctl start speech-translation

# Check status
sudo systemctl status speech-translation
```

### Nginx Reverse Proxy

Create `/etc/nginx/sites-available/speech-translation`:

```nginx
server {
    listen 80;
    server_name your-domain.com www.your-domain.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com www.your-domain.com;
    
    # SSL Configuration
    ssl_certificate /path/to/your/certificate.crt;
    ssl_certificate_key /path/to/your/private.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";
    
    # File upload size
    client_max_body_size 100M;
    
    # Proxy to Streamlit
    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_read_timeout 86400;
    }
    
    # Static files (if any)
    location /static/ {
        alias /opt/speech-translation/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    
    # Health check endpoint
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/speech-translation /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Restart Nginx
sudo systemctl restart nginx
```

---

## Docker Deployment

### Dockerfile

Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p outputs logs uploads models

# Create non-root user
RUN useradd -m -u 1000 speechapp && \
    chown -R speechapp:speechapp /app

# Switch to non-root user
USER speechapp

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/health || exit 1

# Start application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  speech-translation:
    build: .
    ports:
      - "8501:8501"
    environment:
      - APP_ENV=production
      - LOG_LEVEL=INFO
    volumes:
      - ./outputs:/app/outputs
      - ./logs:/app/logs
      - ./uploads:/app/uploads
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - speech-translation
    restart: unless-stopped

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  redis_data:
```

### Build and Deploy

```bash
# Build image
docker build -t speech-translation:latest .

# Run with Docker Compose
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f speech-translation

# Update deployment
docker-compose pull
docker-compose up -d
```

---

## Cloud Deployment

### AWS Deployment

#### EC2 Instance Setup

```bash
# Launch EC2 instance (t3.medium or larger)
# Security Group: Allow HTTP (80), HTTPS (443), SSH (22)

# Connect to instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Follow production deployment steps
# ...
```

#### ECS Deployment

Create `task-definition.json`:

```json
{
  "family": "speech-translation",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "speech-translation",
      "image": "your-account.dkr.ecr.region.amazonaws.com/speech-translation:latest",
      "portMappings": [
        {
          "containerPort": 8501,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "APP_ENV",
          "value": "production"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/speech-translation",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### Azure Deployment

#### Container Instances

```bash
# Create resource group
az group create --name speech-translation-rg --location eastus

# Deploy container
az container create \
  --resource-group speech-translation-rg \
  --name speech-translation \
  --image your-registry/speech-translation:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8501 \
  --environment-variables APP_ENV=production
```

### Google Cloud Platform

#### Cloud Run Deployment

```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/your-project/speech-translation

# Deploy to Cloud Run
gcloud run deploy speech-translation \
  --image gcr.io/your-project/speech-translation \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2
```

---

## Monitoring and Maintenance

### Health Monitoring

Create `health_check.py`:

```python
#!/usr/bin/env python3
import requests
import sys
import time
from datetime import datetime

def check_health(url="http://localhost:8501"):
    """Check application health."""
    try:
        # Check main endpoint
        response = requests.get(f"{url}/health", timeout=10)
        
        if response.status_code == 200:
            print(f"✅ Health check passed at {datetime.now()}")
            return True
        else:
            print(f"❌ Health check failed: HTTP {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed: {e}")
        return False

def main():
    """Main health check function."""
    if not check_health():
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### Log Monitoring

```bash
# Monitor application logs
tail -f logs/pipeline_*.log

# Monitor system logs
sudo journalctl -u speech-translation -f

# Monitor Nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### Performance Monitoring

Create monitoring script `monitor.py`:

```python
#!/usr/bin/env python3
import psutil
import time
import json
from datetime import datetime

def collect_metrics():
    """Collect system metrics."""
    return {
        'timestamp': datetime.now().isoformat(),
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_percent': psutil.disk_usage('/').percent,
        'load_average': psutil.getloadavg(),
        'network_io': psutil.net_io_counters()._asdict()
    }

def main():
    """Main monitoring loop."""
    while True:
        metrics = collect_metrics()
        
        # Log metrics
        with open('logs/metrics.log', 'a') as f:
            f.write(json.dumps(metrics) + '\n')
        
        # Alert on high resource usage
        if metrics['cpu_percent'] > 80:
            print(f"⚠️  High CPU usage: {metrics['cpu_percent']:.1f}%")
        
        if metrics['memory_percent'] > 80:
            print(f"⚠️  High memory usage: {metrics['memory_percent']:.1f}%")
        
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    main()
```

### Automated Backups

```bash
#!/bin/bash
# backup.sh - Automated backup script

BACKUP_DIR="/backup/speech-translation"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Backup application data
tar -czf "$BACKUP_DIR/app_backup_$DATE.tar.gz" \
    /opt/speech-translation/outputs \
    /opt/speech-translation/logs \
    /opt/speech-translation/.env

# Backup configuration
cp /etc/systemd/system/speech-translation.service "$BACKUP_DIR/service_$DATE.conf"
cp /etc/nginx/sites-available/speech-translation "$BACKUP_DIR/nginx_$DATE.conf"

# Clean old backups (keep 7 days)
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +7 -delete

echo "Backup completed: $BACKUP_DIR/app_backup_$DATE.tar.gz"
```

### Cron Jobs

```bash
# Add to crontab (crontab -e)

# Health check every 5 minutes
*/5 * * * * /opt/speech-translation/.venv/bin/python /opt/speech-translation/health_check.py

# Daily backup at 2 AM
0 2 * * * /opt/speech-translation/backup.sh

# Weekly log rotation
0 0 * * 0 /usr/sbin/logrotate /etc/logrotate.d/speech-translation

# Monthly performance report
0 0 1 * * /opt/speech-translation/.venv/bin/python /opt/speech-translation/benchmark_performance.py
```

---

## Security Considerations

### Application Security

1. **Input Validation**
   - File type validation
   - File size limits
   - Content scanning

2. **Access Control**
   - Authentication (if required)
   - Rate limiting
   - IP whitelisting

3. **Data Protection**
   - Secure file storage
   - Data encryption at rest
   - Secure data transmission

### Server Security

```bash
# Update system regularly
sudo apt update && sudo apt upgrade -y

# Configure firewall
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow 80
sudo ufw allow 443

# Disable root login
sudo sed -i 's/PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config
sudo systemctl restart ssh

# Install fail2ban
sudo apt install fail2ban -y
```

### SSL/TLS Configuration

```bash
# Install Certbot for Let's Encrypt
sudo apt install certbot python3-certbot-nginx -y

# Obtain SSL certificate
sudo certbot --nginx -d your-domain.com -d www.your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

---

## Troubleshooting

### Common Deployment Issues

#### Service Won't Start

```bash
# Check service status
sudo systemctl status speech-translation

# Check logs
sudo journalctl -u speech-translation -n 50

# Check Python environment
sudo su - speechapp
source /opt/speech-translation/.venv/bin/activate
python -c "import streamlit; print('OK')"
```

#### High Memory Usage

```bash
# Monitor memory usage
top -p $(pgrep -f streamlit)

# Check for memory leaks
valgrind --tool=memcheck python app.py

# Restart service if needed
sudo systemctl restart speech-translation
```

#### Slow Performance

```bash
# Check system resources
htop
iostat -x 1

# Profile application
python -m cProfile -o profile.stats app.py

# Check disk space
df -h
```

#### Network Issues

```bash
# Check port binding
sudo netstat -tlnp | grep 8501

# Test connectivity
curl -I http://localhost:8501

# Check firewall
sudo ufw status
```

### Recovery Procedures

#### Service Recovery

```bash
# Stop service
sudo systemctl stop speech-translation

# Clear temporary files
sudo rm -rf /opt/speech-translation/outputs/temp/*

# Restart service
sudo systemctl start speech-translation
```

#### Database Recovery (if applicable)

```bash
# Restore from backup
sudo systemctl stop speech-translation
tar -xzf backup/app_backup_YYYYMMDD_HHMMSS.tar.gz -C /
sudo systemctl start speech-translation
```

---

## Scaling and Performance

### Horizontal Scaling

#### Load Balancer Configuration

```nginx
upstream speech_translation {
    server 127.0.0.1:8501;
    server 127.0.0.1:8502;
    server 127.0.0.1:8503;
}

server {
    listen 80;
    
    location / {
        proxy_pass http://speech_translation;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

#### Multiple Instance Deployment

```bash
# Create multiple service files
for i in {1..3}; do
    sudo cp /etc/systemd/system/speech-translation.service \
           /etc/systemd/system/speech-translation-$i.service
    
    # Update port in each service file
    sudo sed -i "s/8501/850$i/g" /etc/systemd/system/speech-translation-$i.service
done

# Start all instances
for i in {1..3}; do
    sudo systemctl enable speech-translation-$i
    sudo systemctl start speech-translation-$i
done
```

### Performance Optimization

#### Caching Strategy

```python
# Redis caching for processed results
import redis
import pickle

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_result(session_id, result):
    """Cache processing result."""
    redis_client.setex(
        f"result:{session_id}",
        3600,  # 1 hour TTL
        pickle.dumps(result)
    )

def get_cached_result(session_id):
    """Get cached result."""
    cached = redis_client.get(f"result:{session_id}")
    if cached:
        return pickle.loads(cached)
    return None
```

#### Resource Optimization

```python
# Memory optimization
import gc

def optimize_memory():
    """Optimize memory usage."""
    gc.collect()  # Force garbage collection
    
# CPU optimization
import multiprocessing

def get_optimal_workers():
    """Get optimal number of workers."""
    return min(multiprocessing.cpu_count(), 4)
```

---

## Conclusion

This deployment guide provides comprehensive instructions for deploying the Speech Translation System in various environments. Key points to remember:

1. **Start Simple**: Begin with local development deployment
2. **Test Thoroughly**: Validate performance and functionality before production
3. **Monitor Continuously**: Implement comprehensive monitoring and alerting
4. **Security First**: Follow security best practices throughout
5. **Plan for Scale**: Design with future scaling requirements in mind

For additional support:
- **API Documentation**: `docs/API_Documentation.md`
- **User Guide**: `docs/User_Guide.md`
- **Performance Benchmarks**: Run `python benchmark_performance.py`

**Deployment Checklist:**
- [ ] System requirements met
- [ ] Dependencies installed
- [ ] Configuration files created
- [ ] Services configured and started
- [ ] Monitoring implemented
- [ ] Security measures in place
- [ ] Backup procedures established
- [ ] Performance validated

Successful deployment! 🚀