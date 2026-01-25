import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from typing import Dict, Any
import asyncio
from .dependencies import get_model
from .schemas import PredictionRequest, PredictionResponse, HealthResponse
from .inference import predict, predict_async
from .metrics import collect_metrics, record_request_metric
from ultralytics import YOLO


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
        # Trigger model loading during startup
        registry = app.dependency_overrides.get(get_model, get_model)()
        next(registry)  # Initialize model
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    yield
    
    # Shutdown: Clean up resources
    logger.info("Shutting down ML Inference Service...")


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
    except Exception as e:
        process_time = time.time() - start_time
        record_request_metric(process_time, is_error=True)
        logger.error(f"Request processing failed: {e}")
        raise


@app.get("/")
async def read_root():
    """Root endpoint for basic service check"""
    return {"message": "ML Inference Platform API", "status": "running"}


@app.get("/health", response_model=HealthResponse)
async def health_check(model: YOLO = Depends(get_model)) -> HealthResponse:
    """Health check endpoint to verify service status"""
    try:
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
    model: YOLO = Depends(get_model)
) -> PredictionResponse:
    """
    Main prediction endpoint for image classification and detection
    """
    try:
        result = predict(model, request)
        return result
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        record_request_metric(time.time() - time.time(), is_error=True)  # Record error
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict_async", response_model=PredictionResponse)
async def predict_async_endpoint(
    request: PredictionRequest,
    model: YOLO = Depends(get_model)
) -> PredictionResponse:
    """
    Async prediction endpoint for image classification and detection
    """
    try:
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
    model: YOLO = Depends(get_model)
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
async def model_info(model: YOLO = Depends(get_model)) -> Dict[str, Any]:
    """Get information about the loaded model"""
    return {
        "model_type": type(model).__name__,
        "model_names": model.names if hasattr(model, 'names') else {},
        "task": getattr(model, 'task', 'unknown'),
        "stride": getattr(model, 'stride', 'unknown')
    }