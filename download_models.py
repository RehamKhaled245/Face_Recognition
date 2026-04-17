import os
from download_models import download_model

os.makedirs("models", exist_ok=True)

def download_all_models():
    download_model("11fRa2g5ql2LKcA6htEYp775mJGq59pgh", "models/shape_predictor_68.dat")
    download_model("1EnIr003y7LRkRAV2QlrWDOlsFYMED5_b", "models/dlib_face_recognition.dat")
    download_model("1KKMY0uGCeJJA43qeXC-Z980P0OxL_ITr", "models/retinaface.caffemodel")
    download_model("1GoRleV0KkvGSjlgriYkUifmFYczdSdg9", "models/deploy.prototxt")

if __name__ == "__main__":
    download_all_models()
    print("Models downloaded successfully")