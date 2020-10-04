import os
import matplotlib.pyplot as plt
import json
import math
from annoy import AnnoyIndex
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from PIL import Image
from numpy import asarray
import numpy as np
import cv2
import ipywidgets as widgets
from IPython.display import display
import io
import matplotlib.pyplot as plt
import imutils

face_detector = MTCNN()
encoder_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
# ann_index = AnnoyIndex(2048, 'angular')
# _ = ann_index.load("celeb_index_60.ann")

def celeb_mapping(celeb_mapping_path):
	with open(celeb_mapping_path) as json_file:
		celeb_mapping_1_temp = json.load(json_file)
	celeb_mapping_1 = {}
	for key, value_list in celeb_mapping_1_temp.items():
		for each_id in value_list:
			celeb_mapping_1[each_id] = str(key)
	return celeb_mapping_1

def get_celeb_name_from_id(result_list, celeb_mapping_path, dist_threshold=0.9):
	id_list = result_list[0]
	dist_list = result_list[1]
	celeb_mapping_dict = celeb_mapping(celeb_mapping_path)
	counts = dict()
	for each_id, each_dist in zip(id_list, dist_list):
		if each_dist < dist_threshold:
			output = celeb_mapping_dict.get(each_id)
			counts[output] = counts.get(output, 0) + 1
	return counts

def face_distance_to_conf(face_distance, face_match_threshold=0.34):
	if face_distance > face_match_threshold:
		range = (1.0 - face_match_threshold)
		linear_val = (1.0 - face_distance) / (range * 2.0)
		return linear_val
	else:
		range = face_match_threshold
		linear_val = 1.0 - (face_distance / (range * 2.0))
		return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))

def get_encoding_new(img):
	results = face_detector.detect_faces(img)
	if len(results)>0:
		encodings = []
		bbox = []
		for result in results:
			x1, y1, width, height = result['box']
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
			encodings.append(encoding)
			bbox.append((x1, y1, width, height))
		return encodings, bbox
	else:
		return None, None

def get_celeb_prediction(img, ann_filepath, celeb_mapping_path):
	ann_index = AnnoyIndex(2048, 'angular')
	_ = ann_index.load(ann_filepath)
	encs, bbox = get_encoding_new(img)
	data = []
	for index, enc in enumerate(encs):
		cv2.rectangle(img, bbox[index], (255,0,0), 2)
		temp_data = {}
		temp_data["bbox"] = bbox[index]
		results = ann_index.get_nns_by_vector(enc[0], 10, search_k=-1, include_distances=True)
		dist_threshold = 0.9
		celeb_count_dict = get_celeb_name_from_id(results, celeb_mapping_path, dist_threshold)
		distance = results[1][0]
		if len(celeb_count_dict)!=0 and max(celeb_count_dict.values()) > 3:
			celeb_name = max(celeb_count_dict, key=celeb_count_dict.get)
			cv2.putText(img, celeb_name.upper(), (bbox[index][0]-5, bbox[index][1] - 5), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 1)
			temp_data["celeb_name"] = celeb_name
			temp_data["confidence"] = face_distance_to_conf(distance)
		else:
			temp_data["celeb_name"] = "unknown"
			temp_data["confidence"] = 0.0
		data.append(temp_data)
	img = imutils.resize(img, width=400)
	# display(img)
	return data, img