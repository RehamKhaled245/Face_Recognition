from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2

from face_service import get_embedding_from_image

app = FastAPI()


@app.post("/embedding")
async def embedding(file: UploadFile = File(...)):
    contents = await file.read()

    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    embeddings = get_embedding_from_image(img)

    if len(embeddings) == 0:
        return {"status": "no face detected"}

    return {"embeddings": embeddings}