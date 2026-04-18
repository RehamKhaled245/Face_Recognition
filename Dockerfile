FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential cmake \
    libopenblas-dev liblapack-dev \
    libx11-dev libgtk-3-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gdown  

COPY . .

# ---- تحميل الموديلز من Google Drive ----
RUN mkdir -p models && \
    gdown "11fRa2g5ql2LKcA6htEYp775mJGq59pgh" -O models/shape_predictor_68_face_landmarks.dat && \
    gdown "1EnIr003y7LRkRAV2QlrWDOlsFYMED5_b" -O models/dlib_face_recognition_resnet_model_v1.dat && \
    gdown "1KKMY0uGCeJJA43qeXC-Z980P0OxL_ITr" -O models/Widerface-RetinaFace.caffemodel && \
    gdown "1GoRleV0KkvGSjlgriYkUifmFYczdSdg9" -O models/deploy.prototxt

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
