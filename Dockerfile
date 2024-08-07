# Use the official Python 3.12 image based on Ubuntu 20.04 as the base image
FROM python:3.12-bullseye

# Set environment variables to ensure Python outputs are sent straight to the terminal
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies required by OpenCV
RUN apt-get update && \
    apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and model into the container
COPY main.py .
COPY models/best.tflite ./models/best.tflite

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application
CMD ["python", "main.py"]
