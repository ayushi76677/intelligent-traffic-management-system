# ==============================================================================
# Smart Traffic Management System — Production Container Image
# ==============================================================================
FROM python:3.11-slim

# Prevent Python from writing .pyc and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000 \
    FLASK_ENV=production

# Install system dependencies (OpenCV headless runtime dependencies)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency definition first for Docker layer caching
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files (excluding patterns in .dockerignore / .gitignore)
COPY . .

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:' + str(os.environ.get('PORT', 5000)) + '/api/status')" || exit 1

CMD ["sh", "-c", "python -m waitress --host=0.0.0.0 --port=${PORT:-5000} --threads=4 app:app"]
