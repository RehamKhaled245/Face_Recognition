import os
import gdown

def download_model(file_id, output_path):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {output_path}...")
        gdown.download(url, output_path, quiet=False)
    else:
        print(f"{output_path} already exists.")