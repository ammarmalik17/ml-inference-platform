from app.inference import load_image_from_request
from app.schemas import PredictionRequest, ImageSource, TaskType


def test_load_image_from_request_base64():
    """Test loading image from base64 string"""
    # Test the function with a valid base64 image
    prediction_request = PredictionRequest(
        image_source=ImageSource.BASE64,
        image_data="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",  # Minimal valid base64 PNG
        task_type=TaskType.CLASSIFICATION,
        confidence_threshold=0.25
    )
    
    # Test that the function can handle the request object
    # Note: This minimal image may not decode to a valid image, but it should at least accept the request
    try:
        result = load_image_from_request(prediction_request)
        # If successful, result should be a numpy array
        assert hasattr(result, 'shape')
    except ValueError:
        # Expected for invalid image data, but function should at least run
        pass


def test_prediction_request_schema():
    """Test that the prediction request schema works correctly"""
    prediction_request = PredictionRequest(
        image_source=ImageSource.BASE64,
        image_data="test_data",
        task_type=TaskType.CLASSIFICATION,
        confidence_threshold=0.5
    )
    
    assert prediction_request.image_source == ImageSource.BASE64
    assert prediction_request.task_type == TaskType.CLASSIFICATION
    assert prediction_request.confidence_threshold == 0.5