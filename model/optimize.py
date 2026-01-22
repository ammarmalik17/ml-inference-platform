def optimize_model(model):
    """Optimize the trained model for inference"""
    print("Optimizing model...")
    # Placeholder for optimization logic
    # return model  // Removed duplicate return


def load_optimized_model(model_path: str = "yolo11n-cls.pt"):
    """Load an optimized model, potentially from various formats"""
    from ultralytics import YOLO
    
    # Check if the model exists locally, otherwise it will be downloaded
    model = YOLO(model_path)
    return model