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
import json

face_detector = MTCNN()
encoder_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
# ann_index = AnnoyIndex(2048, 'angular')

def load_indexes(ann_filepath=None, celeb_mapping_path=None):
	home = expanduser("~")
	if ann_filepath is None:
		ann_filepath = os.path.join(home,'celeb_index_60.ann')
		celeb_ann_id = '1-3Wb7fiINbrk9FSagTxjLdSjp7KzrMp7'
		if not os.path.exists(ann_filepath):
			download_file_from_google_drive(celeb_ann_id, ann_filepath)

	if celeb_mapping_path is None:
		celeb_mapping_path = os.path.join(home,'celeb_mapping.json')
		celeb_mapping_file_id = '1wDaaSQ6NjxLkxpzYyTRknefizZUKnKDj'
		if not os.path.exists(celeb_mapping_path):
			download_file_from_google_drive(celeb_mapping_file_id, celeb_mapping_path)

	ann_index = AnnoyIndex(2048, 'angular')
	_ = ann_index.load(ann_filepath)

	with open(celeb_mapping_path) as json_file:
		celeb_mapping_temp = json.load(json_file)
	celeb_mapping_dict = {}
	for key, value_list in celeb_mapping_temp.items():
		for each_id in value_list:
			celeb_mapping_dict[each_id] = str(key)

	return ann_index, celeb_mapping_dict

def get_encoding(img_path):
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

def save_json(celeb_mapping):
	with open("celeb_mapping.json", "w") as outfile:  
		json.dump(celeb_mapping, outfile)