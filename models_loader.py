import cv2
import dlib

# -------------------------------
# Load Models (ONLY ONCE)
# -------------------------------

sp_68 = dlib.shape_predictor("models/shape_predictor_68.dat")
face_rec_model = dlib.face_recognition_model_v1("models/dlib_face_recognition.dat")

net = cv2.dnn.readNetFromCaffe(
    "models/deploy.prototxt",
    "models/retinaface.caffemodel"
)