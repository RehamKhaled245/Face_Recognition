from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import dlib
import json
import os

app = FastAPI(title="Face Embedding API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Load Models ----
BASE = os.path.dirname(__file__)

sp_68 = dlib.shape_predictor(os.path.join(BASE, "models/shape_predictor_68_face_landmarks.dat"))
face_rec_model = dlib.face_recognition_model_v1(os.path.join(BASE, "models/dlib_face_recognition_resnet_model_v1.dat"))

modelFile = os.path.join(BASE, "models/Widerface-RetinaFace.caffemodel")
configFile = os.path.join(BASE, "models/deploy.prototxt")
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# ---- Load Face Database ----
data_file = os.path.join(BASE, "face_data.json")
if os.path.exists(data_file):
    with open(data_file, "r") as f:
        face_db = json.load(f)
else:
    face_db = {}

# ---- Helper Functions ----
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
    return np.array(emb)

def euclidean_distance(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))

def find_best_match(query_emb, threshold=0.5):
    best_name = None
    best_distance = float("inf")
    all_results = []

    for name, embeddings in face_db.items():
        distances = [euclidean_distance(query_emb, emb) for emb in embeddings]
        min_dist = min(distances)
        all_results.append({"name": name, "distance": round(min_dist, 4)})
        if min_dist < best_distance:
            best_distance = min_dist
            best_name = name

    all_results.sort(key=lambda x: x["distance"])

    verified = best_distance < threshold
    return {
        "verified": verified,
        "name": best_name if verified else "Unknown",
        "distance": round(best_distance, 4),
        "threshold": threshold,
        "all_matches": all_results[:5]
    }

# ---- Endpoints ----

@app.post("/embed")
async def embed_face(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    faces = detect_faces(img_bgr)

    if len(faces) == 0:
        raise HTTPException(status_code=404, detail="No face detected in image")

    embeddings = []
    for rect in faces:
        emb = get_embedding(img_rgb, rect)
        embeddings.append({
            "bbox": {
                "x1": rect.left(), "y1": rect.top(),
                "x2": rect.right(), "y2": rect.bottom()
            },
            "embedding": emb.tolist(),
            "embedding_size": len(emb)
        })

    return JSONResponse({"faces_found": len(embeddings), "results": embeddings})


@app.post("/verify")
async def verify_face(file: UploadFile = File(...), threshold: float = 0.5):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    faces = detect_faces(img_bgr)

    if len(faces) == 0:
        raise HTTPException(status_code=404, detail="No face detected in image")

    if not face_db:
        raise HTTPException(status_code=404, detail="Face database is empty")

    results = []
    for rect in faces:
        emb = get_embedding(img_rgb, rect)
        match = find_best_match(emb, threshold)
        match["bbox"] = {
            "x1": rect.left(), "y1": rect.top(),
            "x2": rect.right(), "y2": rect.bottom()
        }
        results.append(match)

    return JSONResponse({
        "faces_found": len(results),
        "results": results
    })


@app.get("/database")
def get_database():
    return JSONResponse({
        "total_persons": len(face_db),
        "persons": [
            {"name": name, "images_count": len(embeddings)}
            for name, embeddings in face_db.items()
        ]
    })


@app.get("/health")
def health():
    return {"status": "ok"}
