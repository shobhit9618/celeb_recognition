import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from mtcnn.mtcnn import MTCNN
import cv2
from PIL import Image
from matplotlib import pyplot
from numpy import asarray
import numpy as np
from os import listdir
import pickle
import json
from annoy import AnnoyIndex
from tqdm import tqdm
from celeb_detector.create_model_utils import get_encoding, save_json 

def create_celeb_model(base_url):
	ann_index = AnnoyIndex(2048, 'angular')
	os.makedirs('celeb_encodings', exist_ok=True)

	celeb_mapping = {}
	c = 0
	print("Starting face detection and encoding creation")
	for folder in os.listdir(base_url):
		celeb_encoding = {}
		celeb_mapping[folder] = []
		for image in tqdm(listdir(base_url + '/' + folder)):
			try:
				encoding = get_encoding(os.path.join(base_url, folder, image))
			except Exception as e:
				print(e)
				continue

			if encoding is not None:
				c += 1
				celeb_encoding[c] = encoding[0]
				celeb_mapping[folder].append(c)
				ann_index.add_item(c, encoding[0])
		save_json(celeb_mapping)
		pickle.dump(celeb_encoding, open(f"celeb_encodings/{folder}_encoding.pkl", "wb" ))
		del celeb_encoding

	save_json(celeb_mapping)
	print("Encoding and mapping files saved successfully")

	print("Building ann index...")
	ann_index.build(1000)
	x = ann_index.save("celeb_index.ann")
	if x:
		print("Ann index saved successfully")
	else:
		print("Error in saving ann index")
