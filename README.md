# ML Inference Platform

A scalable machine learning inference platform built with FastAPI and YOLO models.

## Features

- High-performance inference API supporting both classification and detection
- Model versioning with support for multiple YOLO model variants
- Scalable architecture with Kubernetes deployment options
- Comprehensive metrics and monitoring
- Thread-safe model loading with singleton pattern
- Support for various image input sources (URL, base64, file upload)

## Prerequisites

- Python 3.8+
- pip

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the development server:
```bash
python run_server.py --reload
```

Or using uvicorn directly:
```bash
uvicorn app.main:app --reload --port 8000
```

3. The API will be available at `http://localhost:8000`

## API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `GET /metrics` - Service metrics
- `POST /predict` - Main prediction endpoint
- `POST /predict_async` - Async prediction endpoint
- `POST /predict_file` - File upload prediction endpoint
- `GET /model/info` - Model information

## Usage Examples

### Classification Request
```json
{
  "image_source": "base64",
  "image_data": "<base64_encoded_image>",
  "task_type": "classification",
  "confidence_threshold": 0.5
}
```

### Detection Request
```json
{
  "image_source": "url",
  "image_data": "https://example.com/image.jpg",
  "task_type": "detection",
  "confidence_threshold": 0.3
}
```

## Docker Deployment

Build and run with Docker:
```bash
docker build -t ml-inference-platform -f docker/Dockerfile .
docker run -p 8000:8000 ml-inference-platform
```

## Kubernetes Deployment

Apply the Kubernetes manifests:
```bash
kubectl apply -f k8s/
```

## Configuration

The service can be configured using environment variables:

- `MODEL_PATH`: Path to the YOLO model file (default: "yolo11n-cls.pt")

## Development

For development, install dev dependencies:
```bash
pip install -r requirements-dev.txt
```

Run tests:
```bash
pytest tests/
```

Format code:
```bash
black .
```