from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    DETECTION = "detection"


class ImageSource(str, Enum):
    URL = "url"
    BASE64 = "base64"
    UPLOAD = "upload"


class ClassificationResult(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)


class DetectionResult(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: Optional[List[float]] = Field(default=None, description="Bounding box [x1, y1, x2, y2]")


class PredictionRequest(BaseModel):
    image_source: ImageSource = Field(default=ImageSource.BASE64, description="Type of image source")
    image_data: str = Field(..., description="Image data as URL, base64 string, or file content")
    task_type: TaskType = Field(default=TaskType.CLASSIFICATION, description="Type of inference task")
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0, description="Confidence threshold for detections")


class PredictionResponse(BaseModel):
    task_type: TaskType
    results: List[ClassificationResult] | List[DetectionResult]
    processing_time: float
    model_version: str


class HealthResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool
    model_version: Optional[str] = None