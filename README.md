# ML Inference Platform

A pure ML inference platform built with FastAPI and YOLO models, optimized for production model serving.

## Features

- High-performance inference API supporting both classification and detection
- Advanced model management with support for multiple concurrent models
- Model loading, unloading, and switching capabilities
- Scalable architecture with Kubernetes deployment options
- Comprehensive metrics and monitoring
- Thread-safe model caching with LRU eviction
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
- `GET /health` - Health check (accepts optional model_path parameter)
- `GET /metrics` - Service metrics
- `POST /predict` - Main prediction endpoint (accepts optional model_path parameter)
- `POST /predict_async` - Async prediction endpoint (accepts optional model_path parameter)
- `POST /predict_file` - File upload prediction endpoint (accepts optional model_path parameter)
- `GET /model/info` - Model information (accepts optional model_path parameter)
- `GET /models` - List all currently loaded models
- `POST /models/load/{model_name}` - Load a specific model into memory
- `DELETE /models/unload/{model_name}` - Unload a specific model from memory

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

### Using Specific Model
To use a specific model, add the model_path parameter as a query parameter:

`POST /predict?model_path=yolo11n.pt`

### Managing Models
- List loaded models: `GET /models`
- Load a model: `POST /models/load/yolo11n-cls.pt`
- Unload a model: `DELETE /models/unload/yolo11n-cls.pt`

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

- `MODEL_PATH`: Default path to the YOLO model file (default: "yolo11n-cls.pt")

For production deployments, you can load models dynamically via the model management API endpoints.

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