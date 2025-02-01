# Use Python 3.9 as base image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy all files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn numpy opencv-python-headless \
    facenet-pytorch torch torchvision torchaudio pillow pytesseract nest-asyncio

# Expose port 7000
EXPOSE 7000

# Command to run FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7000"]
