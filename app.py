from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import dlib
import io
from PIL import Image

app = FastAPI(title="Face Embedding API")

# ---- Load Models (مرة واحدة عند بدء السيرفر) ----
import os
BASE = os.path.dirname(__file__)

sp_68 = dlib.shape_predictor(os.path.join(BASE, "models/shape_predictor_68_face_landmarks.dat"))
face_rec_model = dlib.face_recognition_model_v1(os.path.join(BASE, "models/dlib_face_recognition_resnet_model_v1.dat"))

modelFile = os.path.join(BASE, "models/Widerface-RetinaFace.caffemodel")
configFile = os.path.join(BASE, "models/deploy.prototxt")
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)


def detect_faces(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()

    rects = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            rects.append(dlib.rectangle(x1, y1, x2, y2))
    return rects


def get_embedding(img_rgb, rect):
    shape = sp_68(img_rgb, rect)
    emb = face_rec_model.compute_face_descriptor(img_rgb, shape)
    return np.array(emb).tolist()


# ---- Endpoint الرئيسي ----
@app.post("/embed")
async def embed_face(file: UploadFile = File(...)):
    # قراءة الصورة
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # كشف الوجوه
    faces = detect_faces(img_bgr)
    if len(faces) == 0:
        raise HTTPException(status_code=404, detail="No face detected in image")

    # استخراج embeddings لكل الوجوه في الصورة
    embeddings = []
    for rect in faces:
        emb = get_embedding(img_rgb, rect)
        embeddings.append({
            "bbox": {
                "x1": rect.left(), "y1": rect.top(),
                "x2": rect.right(), "y2": rect.bottom()
            },
            "embedding": emb,
            "embedding_size": len(emb)
        })

    return JSONResponse({
        "faces_found": len(embeddings),
        "results": embeddings
    })


@app.get("/health")
def health():
    return {"status": "ok"}