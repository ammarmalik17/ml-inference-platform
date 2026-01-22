import pytest
from fastapi.testclient import TestClient
from app.main import app


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