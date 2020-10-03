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

def get_encoding(img_path, face_detector, encoder_model):
	img = pyplot.imread(img_path)
	results = face_detector.detect_faces(img)
	if len(results)>0:
		x1, y1, width, height = results[0]['box']

		if x1 <0:
			x1 = 0
		if y1 <0:
			y1 = 0
		
		x2, y2 = x1 + width, y1 + height
		face = img[y1:y2, x1:x2]
		image = Image.fromarray(face)
		image = image.resize((224,224))
		face_array = asarray(image)

		samples = asarray(face_array, 'float32')
		samples = preprocess_input(samples, version=2)
		samples = np.expand_dims(samples, axis=0)
		encoding = encoder_model.predict(samples)

		return encoding
	else:
		return None

face_detector = MTCNN()
encoder_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
ann_index = AnnoyIndex(2048, 'angular')

os.makedirs('celeb_encodings', exist_ok=True)

celeb_mapping = {}
c = 0
def save_json(celeb_mapping):
	with open("celeb_mapping.json", "w") as outfile:  
		json.dump(celeb_mapping, outfile) 

#provide path to images directory here (refer README for directory structure)
base_url = 'provide name of celeb images dir'

print("Starting loop")
for folder in os.listdir(base_url):
	celeb_encoding = {}
	celeb_mapping[folder] = []
	print(f"folder {folder}")
	for image in listdir(base_url + '/' + folder):
		c += 1
		if c%100==0:
			print(str(c))
		encoding = get_encoding(base_url + '/' + folder + '/' + image, face_detector, encoder_model)
		if encoding is not None:
			celeb_encoding[c] = encoding[0]
			celeb_mapping[folder].append(c)
			ann_index.add_item(c, encoding[0])
	save_json(celeb_mapping)
	pickle.dump(celeb_encoding, open(f"celeb_encodings/{folder}_encoding.pkl", "wb" ))
	del celeb_encoding

print("Building ann index...")
ann_index.build(1000)
x = ann_index.save("celeb_index.ann")
if x:
	print("Ann index saved successfully")
else:
	print("Error in saving ann index")
