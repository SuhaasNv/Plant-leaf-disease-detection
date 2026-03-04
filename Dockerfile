FROM python:3.10-slim

WORKDIR /app

# libgomp1: required by TensorFlow (OpenMP threading)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# config.py must be in the same directory as main.py so `import config` resolves
COPY config.py .
COPY api/main.py .

# Model file (from project root; add trained_plant_disease_model.h5 to repo)
COPY trained_plant_disease_model.h5 .

EXPOSE 8080

# Railway injects PORT (default 8080); use it so the app is reachable
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
