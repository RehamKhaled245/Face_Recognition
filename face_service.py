import cv2
import numpy as np
import dlib
from models_loader import sp_68, face_rec_model, net


# -------------------------------
# Face Detection
# -------------------------------

def detect_faces(frame):
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
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


# -------------------------------
# Embedding Extraction
# -------------------------------

def get_embedding_from_image(img):
    faces = detect_faces(img)

    if len(faces) == 0:
        return []

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    embeddings = []

    for rect in faces:
        shape = sp_68(img_rgb, rect)
        emb = face_rec_model.compute_face_descriptor(img_rgb, shape)
        embeddings.append(np.array(emb).tolist())

    return embeddings