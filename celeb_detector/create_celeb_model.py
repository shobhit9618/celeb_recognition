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
    counter = 0
    
    print("Starting face detection and encoding creation")
    
    # Process images in batches
    batch_size = 32
    for folder in os.listdir(base_path):
        celeb_mapping[folder] = []
        image_paths = [os.path.join(base_path, folder, img) 
                      for img in os.listdir(os.path.join(base_path, folder))]
        
        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i:i + batch_size]
            for image_path in batch_paths:
                try:
                    image = cv2.imread(image_path)
                    encodings, bboxes = celeb_recog.get_encoding(image)
                    if encodings:
                        for encoding, bbox in zip(encodings, bboxes):
                            counter += 1
                            celeb_encoding[counter] = encoding.squeeze()
                            celeb_mapping[folder].append(counter)
                            ann_index.add_item(counter, encoding.squeeze())
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    continue
            
            # Save batch results
            if celeb_encoding:
                with open(f"celeb_encodings/{folder}_encoding_{i}.pkl", "wb") as f:
                    pickle.dump(celeb_encoding, f)
                celeb_encoding.clear()

    print("Building ann index...")
    ann_index.build(1000, n_jobs=-1)  # Use all available cores
    if ann_index.save("celeb_index.ann"):
        print("Ann index saved successfully")
    else:
        print("Error in saving ann index")

    save_json(celeb_mapping)

if __name__ == "__main__":
	create_model = create_celeb_model("")
