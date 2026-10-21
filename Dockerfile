FROM python:3.10-slim

# System dependencies for OpenCV & MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Dependencies install karein
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Application code copy karein
COPY main.py /code/main.py

# Cloud Run ka port dynamic rakhein
ENV PORT=8080

CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}
