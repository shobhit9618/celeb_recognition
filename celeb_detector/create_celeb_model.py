import os
import cv2
import numpy as np
import pickle
import json
from annoy import AnnoyIndex
from tqdm import tqdm
import string, random
from celeb_detector.celeb_prediction_main import CelebRecognition

celeb_recog = CelebRecognition()

def generate_random_string(length):
    pool = string.ascii_letters + string.digits
    return ''.join([random.choice(pool) for _ in range(length)])

def save_json(celeb_mapping):
	with open(f"celeb_mapping_{generate_random_string(5)}.json", "w") as outfile:  
		json.dump(celeb_mapping, outfile)

def create_celeb_model(base_path):
    ann_index = AnnoyIndex(2048, 'angular')
    os.makedirs('celeb_encodings', exist_ok=True)
    celeb_mapping = {}
    celeb_encoding = {}
    c = 0
    print("Starting face detection and encoding creation")
    for folder in os.listdir(base_path):
        celeb_mapping[folder] = []
        for image_name in tqdm(os.listdir(os.path.join(base_path, folder))):
            image = cv2.imread(os.path.join(base_path, folder, image_name))
            try:
                encodings, bboxes = celeb_recog.get_encoding(image)
            except Exception as e:
                print(e)
                continue
            if encodings is not None:
                for encoding, bbox in zip(encodings, bboxes):
                    c += 1
                    celeb_encoding[c] = encoding.squeeze()
                    celeb_mapping[folder].append(c)
                    ann_index.add_item(c, encoding.squeeze())
            with open(f"celeb_encodings/{folder}_encoding.pkl", "wb") as f:
                pickle.dump(celeb_encoding, f)
            celeb_encoding.clear()
    print("Encoding and mapping files saved successfully")
    print("Building ann index...")
    ann_index.build(1000)
    x = ann_index.save("celeb_index.ann")
    if x:
        print("Ann index saved successfully")
    else:
        print("Error in saving ann index")

    save_json(celeb_mapping)

if __name__ == "__main__":
	create_model = create_celeb_model("")
