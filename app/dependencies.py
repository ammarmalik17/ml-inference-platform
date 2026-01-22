import os
from typing import Generator
from ultralytics import YOLO


class ModelRegistry:
    """Singleton class to manage model instances"""
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
        return cls._instance
    
    def get_model(self):
        """Get or create the YOLO model instance"""
        if self._model is None:
            # Load the smallest YOLO model for classification
            model_path = os.getenv("MODEL_PATH", "yolo11n-cls.pt")
            self._model = YOLO(model_path)
        return self._model


def get_model() -> Generator[YOLO, None, None]:
    """Dependency to load and return the YOLO model"""
    registry = ModelRegistry()
    model = registry.get_model()
    yield model