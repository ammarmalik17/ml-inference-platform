import time
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, File, UploadFile, Query
from fastapi.responses import JSONResponse
from typing import Dict, Any
from .dependencies import ModelRegistry
from .schemas import PredictionRequest, PredictionResponse, HealthResponse
from .inference import predict, predict_async
from .metrics import collect_metrics, record_request_metric


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for FastAPI application
    Used for startup and shutdown events
    """
    logger.info("Starting ML Inference Service...")
    # Startup: Load model, initialize resources
    try:
        # Initialize model registry during startup
        registry = ModelRegistry()
        model_path = os.getenv("MODEL_PATH", "yolo11n-cls.pt")
        registry.get_model(model_path)  # Initialize default model
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    yield
    
    # Shutdown: Clean up resources
    logger.info("Shutting down ML Inference Service...")
    try:
        registry = ModelRegistry()
        registry.clear_cache()  # Clear all loaded models
        logger.info("Models cleared successfully")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


app = FastAPI(
    title="ML Inference Platform API",
    description="A scalable ML inference platform using YOLO models",
    version="1.0.0",
    lifespan=lifespan
)


@app.middleware("http")
async def add_process_time_header(request, call_next):
    """Middleware to measure request processing time and record metrics"""
    start_time = time.time()
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        # Record metrics (excluding health check endpoint)
        if request.url.path != "/health":
            record_request_metric(process_time, is_error=response.status_code >= 400)
        response.headers["X-Process-Time"] = str(process_time)
        return response
    except Exception:
        process_time = time.time() - start_time
        record_request_metric(process_time, is_error=True)
        raise


@app.get("/")
async def read_root():
    """Root endpoint for basic service check"""
    return {"message": "ML Inference Platform API", "status": "running"}


@app.get("/health", response_model=HealthResponse)
async def health_check(model_path: str = Query(None, description="Path to the model to check health for")) -> HealthResponse:
    """Health check endpoint to verify service status"""
    try:
        # Get model based on path or default
        registry = ModelRegistry()
        model = registry.get_model(model_path)
        # Check if model is loaded and accessible
        model_loaded = model is not None
        model_version = getattr(model, 'names', {}).get(0, 'unknown') if model_loaded else None
        
        return HealthResponse(
            status="healthy",
            model_loaded=model_loaded,
            model_version=model_version
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False
        )


@app.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """Endpoint to retrieve service metrics"""
    return collect_metrics()


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(
    request: PredictionRequest,
    model_path: str = Query(None, description="Path to the model to use for prediction")
) -> PredictionResponse:
    """
    Main prediction endpoint for image classification and detection
    """
    try:
        # Get model based on path or default
        registry = ModelRegistry()
        model = registry.get_model(model_path)
        result = predict(model, request)
        return result
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        record_request_metric(time.time() - time.time(), is_error=True)  # Record error
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict_async", response_model=PredictionResponse)
async def predict_async_endpoint(
    request: PredictionRequest,
    model_path: str = Query(None, description="Path to the model to use for prediction")
) -> PredictionResponse:
    """
    Async prediction endpoint for image classification and detection
    """
    try:
        # Get model based on path or default
        registry = ModelRegistry()
        model = registry.get_model(model_path)
        result = await predict_async(model, request)
        return result
    except Exception as e:
        logger.error(f"Async prediction failed: {e}")
        record_request_metric(time.time() - time.time(), is_error=True)  # Record error
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict_file", response_model=PredictionResponse)
async def predict_file_endpoint(
    file: UploadFile = File(...),
    task_type: str = "classification",
    confidence_threshold: float = 0.25,
    model_path: str = Query(None, description="Path to the model to use for prediction")
) -> PredictionResponse:
    """
    Prediction endpoint that accepts uploaded image files
    """
    try:
        # Read file content
        contents = await file.read()
        
        # Create a request object from the uploaded file
        from .schemas import ImageSource, TaskType
        request = PredictionRequest(
            image_source=ImageSource.UPLOAD,
            image_data=contents.hex(),  # Convert bytes to hex string for storage
            task_type=TaskType(task_type.lower()),
            confidence_threshold=confidence_threshold
        )
        
        # Get model based on path or default
        registry = ModelRegistry()
        model = registry.get_model(model_path)
        result = predict(model, request)
        return result
    except Exception as e:
        logger.error(f"File prediction failed: {e}")
        record_request_metric(time.time() - time.time(), is_error=True)  # Record error
        raise HTTPException(status_code=400, detail=str(e))


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler for the application"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred"}
    )


# Additional utility endpoints
@app.get("/model/info")
async def model_info(model_path: str = Query(None, description="Path to the model to get info for")) -> Dict[str, Any]:
    """Get information about the currently active model"""
    # Get model based on path or default
    registry = ModelRegistry()
    model = registry.get_model(model_path)
    return {
        "model_type": type(model).__name__,
        "model_names": model.names if hasattr(model, 'names') else {},
        "task": getattr(model, 'task', 'unknown'),
        "stride": getattr(model, 'stride', 'unknown')
    }


@app.get("/models")
async def list_models() -> Dict[str, Any]:
    """List all currently loaded models"""
    registry = ModelRegistry()
    loaded_models = registry.list_models()
    return {
        "loaded_models": loaded_models,
        "count": len(loaded_models)
    }


@app.post("/models/load/{model_name}")
async def load_model(model_name: str) -> Dict[str, Any]:
    """Load a specific model into memory"""
    try:
        registry = ModelRegistry()
        registry.load_model(model_name)
        return {
            "status": "success",
            "model_loaded": model_name,
            "message": f"Model {model_name} loaded successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load model {model_name}: {str(e)}")


@app.delete("/models/unload/{model_name}")
async def unload_model(model_name: str) -> Dict[str, Any]:
    """Unload a specific model from memory"""
    registry = ModelRegistry()
    success = registry.unload_model(model_name)
    if success:
        return {
            "status": "success",
            "model_unloaded": model_name,
            "message": f"Model {model_name} unloaded successfully"
        }
    else:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found in memory")