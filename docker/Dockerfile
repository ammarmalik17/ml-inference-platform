FROM python:3.9-slim

# Install system dependencies for OpenCV and other computer vision libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create model_files directory and set permissions
RUN mkdir -p model_files

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Set environment variable for model path
ENV MODEL_PATH=model_files/yolo11n-cls.pt

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]