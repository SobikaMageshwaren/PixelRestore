# ==========================================================
# VisionRestore AI Docker Image
# ==========================================================

FROM python:3.11-slim

# ----------------------------------------------------------
# Environment Variables
# ----------------------------------------------------------

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ----------------------------------------------------------
# Install System Dependencies
# ----------------------------------------------------------

RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------
# Set Working Directory
# ----------------------------------------------------------

WORKDIR /app

# ----------------------------------------------------------
# Copy Requirements
# ----------------------------------------------------------

COPY requirements.txt .

# ----------------------------------------------------------
# Install Python Packages
# ----------------------------------------------------------

RUN pip install --upgrade pip

RUN pip install --upgrade pip

RUN pip install --no-cache-dir \
    torch==2.7.1 \
    torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt
# ----------------------------------------------------------
# Copy Project
# ----------------------------------------------------------

COPY . .

# ----------------------------------------------------------
# Create Output Directories
# ----------------------------------------------------------

RUN mkdir -p outputs reports

# ----------------------------------------------------------
# Default Command
# ----------------------------------------------------------

CMD ["python", "main.py"]