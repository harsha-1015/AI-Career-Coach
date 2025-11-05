# ─────────────────────────────────────────────
# Base image: Python 3.13 (slim = smaller image)
# ─────────────────────────────────────────────
FROM python:3.13-slim


RUN apt update -y && apt install awscli -y
# Set working directory inside the container
WORKDIR /app

# ─────────────────────────────────────────────
# Install required system packages for OpenCV, DeepFace & uv
# ─────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────
# Copy project files into container
# ─────────────────────────────────────────────
COPY . .

# ─────────────────────────────────────────────
# Install uv (modern package manager)
# ─────────────────────────────────────────────
RUN pip install --no-cache-dir uv

# Install dependencies from pyproject.toml
RUN uv sync --frozen

# ─────────────────────────────────────────────
# Environment configuration
# ─────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PORT=5000 \
    CUDA_VISIBLE_DEVICES=-1   

# Expose Flask port
EXPOSE 8000

# ─────────────────────────────────────────────
# Start Flask app using uv
# ─────────────────────────────────────────────
CMD ["uv", "run", "main.py"]
