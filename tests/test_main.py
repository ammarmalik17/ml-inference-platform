import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.schemas import PredictionRequest, ImageSource, TaskType


@pytest.fixture
def client():
    """Create a test client for the FastAPI app"""
    with TestClient(app) as test_client:
        yield test_client


def test_read_root(client):
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert response.json()["status"] == "running"


def test_health_endpoint(client):
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    # Note: This test may fail if model isn't properly loaded in test environment


def test_metrics_endpoint(client):
    """Test the metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "requests_count" in response.json()


def test_model_info_endpoint(client):
    """Test the model info endpoint"""
    response = client.get("/model/info")
    assert response.status_code == 200
    assert "model_type" in response.json()


def test_list_models_endpoint(client):
    """Test the list models endpoint"""
    response = client.get("/models")
    assert response.status_code == 200
    assert "loaded_models" in response.json()
    assert "count" in response.json()


def test_prediction_request_schema_validation():
    """Test that the prediction request schema validation works"""
    # Test valid request
    request = PredictionRequest(
        image_source=ImageSource.BASE64,
        image_data="test_data",
        task_type=TaskType.CLASSIFICATION,
        confidence_threshold=0.5
    )
    assert request.image_source == ImageSource.BASE64
    assert request.task_type == TaskType.CLASSIFICATION
    assert request.confidence_threshold == 0.5


def test_api_docs_available(client):
    """Test that API documentation endpoints are available"""
    response = client.get("/docs")
    assert response.status_code in [200, 307]  # May redirect to /docs/ or return directly
    
    response = client.get("/redoc")
    assert response.status_code in [200, 307]