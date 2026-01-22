import os
from typing import Generator
from ultralytics import YOLO
from .optimize import optimize_model_for_inference, quantize_model_for_inference


import threading
from collections import OrderedDict


class ModelRegistry:
    """Singleton class to manage multiple model instances for inference"""
    _instance = None
    _models = None
    _lock = threading.Lock()
    _loaded_models_limit = 5  # Limit number of concurrently loaded models
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._models is None:
            self._models = OrderedDict()
    
    def get_model(self, model_path: str = None):
        """Get or create a YOLO model instance by path"""
        if model_path is None:
            model_path = os.getenv("MODEL_PATH", "yolo11n-cls.pt")
        
        with self._lock:
            if model_path in self._models:
                # Move to end to mark as recently used
                self._models.move_to_end(model_path)
                return self._models[model_path]
            else:
                # Load new model with optimization for inference
                model = optimize_model_for_inference(model_path)
                self._models[model_path] = model
                
                # If we exceed the limit, remove the oldest model
                if len(self._models) > self._loaded_models_limit:
                    oldest_key = next(iter(self._models))
                    del self._models[oldest_key]
                
                return model
    
    def list_models(self):
        """List all loaded models"""
        with self._lock:
            return list(self._models.keys())
    
    def load_model(self, model_path: str):
        """Explicitly load a model"""
        return self.get_model(model_path)
    
    def unload_model(self, model_path: str):
        """Unload a specific model"""
        with self._lock:
            if model_path in self._models:
                del self._models[model_path]
                return True
            return False
    
    def clear_cache(self):
        """Clear all loaded models"""
        with self._lock:
            self._models.clear()


def get_model(model_path: str = None) -> Generator[YOLO, None, None]:
    """Dependency to load and return the YOLO model"""
    registry = ModelRegistry()
    model = registry.get_model(model_path)
    yield model