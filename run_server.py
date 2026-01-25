#!/usr/bin/env python3
"""
Run the ML Inference Platform server
"""
import uvicorn
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="ML Inference Platform Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code changes")
    
    args = parser.parse_args()
    
    # Set environment variable for model path if not set
    if not os.environ.get("MODEL_PATH"):
        os.environ["MODEL_PATH"] = "model_files/yolo11n-cls.pt"  # Default to smallest YOLO model in model_files directory
    
    print(f"Starting ML Inference Platform on {args.host}:{args.port}")
    print(f"Using model: {os.environ['MODEL_PATH']}")
    
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()