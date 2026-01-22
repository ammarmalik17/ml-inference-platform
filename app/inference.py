import time
import base64
import io
from typing import Union, List
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image
import requests
from urllib.parse import urlparse
from .schemas import (
    PredictionRequest, 
    PredictionResponse, 
    TaskType, 
    ClassificationResult, 
    DetectionResult,
    ImageSource
)


def load_image_from_request(request: PredictionRequest) -> np.ndarray:
    """
    Load image from various sources based on the request type
    """
    if request.image_source == ImageSource.URL:
        # Download image from URL
        response = requests.get(request.image_data)
        response.raise_for_status()
        image_bytes = io.BytesIO(response.content)
        image = Image.open(image_bytes)
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    elif request.image_source == ImageSource.BASE64:
        # Decode base64 image
        image_data = base64.b64decode(request.image_data)
        image_array = np.frombuffer(image_data, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Invalid base64 image data")
        return image
    
    elif request.image_source == ImageSource.UPLOAD:
        # For uploaded files, we expect hex string representation of bytes
        try:
            image_bytes = bytes.fromhex(request.image_data)
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Invalid image data")
            return image
        except ValueError:
            # If it's not hex, try treating it as raw bytes
            image_array = np.frombuffer(request.image_data.encode(), dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Invalid image data")
            return image
    
    else:
        raise ValueError(f"Unsupported image source: {request.image_source}")


def perform_classification(model: YOLO, image: np.ndarray, confidence_threshold: float) -> List[ClassificationResult]:
    """
    Perform image classification using the YOLO model
    """
    # Run inference
    results = model(image, verbose=False)
    
    # Extract classification results
    classification_results = []
    
    if hasattr(results[0], 'probs') and results[0].probs is not None:
        # Get top predictions
        top_indices = results[0].probs.top5
        top_scores = results[0].probs.top5conf
        
        for idx, conf in zip(top_indices, top_scores):
            if conf >= confidence_threshold:
                class_name = model.names[int(idx)]
                classification_results.append(
                    ClassificationResult(
                        class_id=int(idx),
                        class_name=class_name,
                        confidence=float(conf)
                    )
                )
    
    return classification_results


def perform_detection(model: YOLO, image: np.ndarray, confidence_threshold: float) -> List[DetectionResult]:
    """
    Perform object detection using the YOLO model
    """
    # Run inference
    results = model(image, verbose=False, conf=confidence_threshold)
    
    # Extract detection results
    detection_results = []
    
    if results[0].boxes is not None:
        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf >= confidence_threshold:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                # Extract bounding box coordinates [x1, y1, x2, y2]
                bbox = box.xyxy[0].tolist()
                
                detection_results.append(
                    DetectionResult(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=conf,
                        bbox=bbox
                    )
                )
    
    return detection_results


def predict(model: YOLO, request: PredictionRequest) -> PredictionResponse:
    """
    Perform inference on the input data
    """
    start_time = time.time()
    
    try:
        # Load image from request
        image = load_image_from_request(request)
        
        # Perform the appropriate type of inference
        if request.task_type == TaskType.CLASSIFICATION:
            results = perform_classification(model, image, request.confidence_threshold)
        elif request.task_type == TaskType.DETECTION:
            results = perform_detection(model, image, request.confidence_threshold)
        else:
            raise ValueError(f"Unsupported task type: {request.task_type}")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Get model version/info
        model_version = getattr(model, 'names', {}).get(0, 'unknown')  # Simplified approach
        
        return PredictionResponse(
            task_type=request.task_type,
            results=results,
            processing_time=processing_time,
            model_version=model_version
        )
    
    except Exception as e:
        processing_time = time.time() - start_time
        raise e


async def predict_async(model: YOLO, request: PredictionRequest) -> PredictionResponse:
    """
    Async version of predict function
    """
    return predict(model, request)