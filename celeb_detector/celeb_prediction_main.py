import pathlib
import torch
import json
import math
from annoy import AnnoyIndex
import face_recognition
from PIL import Image
import numpy as np
import cv2
import imutils

class CelebRecognition:
	def __init__(self):
		file_path = pathlib.Path(__file__).parent.absolute()
		self.celeb_mapping_filepath = f"{file_path}/models/celeb_mapping_117.json"
		celeb_index_annpath = f"{file_path}/models/celeb_index_117.ann"
		vggface_modelpath = f"{file_path}/models/vggface_resnet50.pt"

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		if torch.backends.mps.is_available():
			self.device = torch.device('mps')

		self.encoder_model = torch.load(vggface_modelpath).to(self.device).eval()
		self.ann_index = AnnoyIndex(2048, 'angular')
		_ = self.ann_index.load(celeb_index_annpath)

	def get_celeb_mapping(self):
		with open(self.celeb_mapping_filepath) as json_file:
			celeb_mapping_temp = json.load(json_file)
		celeb_mapping = {}
		for key, value_list in celeb_mapping_temp.items():
			for each_id in value_list:
				celeb_mapping[each_id] = str(key)
		return celeb_mapping

	def get_celeb_name_from_id(self, result_list, dist_threshold=0.9):
		id_list = result_list[0]
		dist_list = result_list[1]
		celeb_mapping_dict = self.get_celeb_mapping()
		counts = dict()
		for each_id, each_dist in zip(id_list, dist_list):
			if each_dist < dist_threshold:
				output = celeb_mapping_dict.get(each_id)
				counts[output] = counts.get(output, 0) + 1
		return counts

	def face_distance_to_conf(self, face_distance, face_match_threshold=0.34):
		if face_distance > face_match_threshold:
			range = (1.0 - face_match_threshold)
			linear_val = (1.0 - face_distance) / (range * 2.0)
			return linear_val
		else:
			range = face_match_threshold
			linear_val = 1.0 - (face_distance / (range * 2.0))
			return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))

	def get_encoding(self, img):
		results = face_recognition.face_locations(img)
		if len(results)>0:
			encodings = []
			bbox = []
			for result in results:
				y1, x2, y2, x1 = result
				if x1 <0:
					x1 = 0
				if y1 <0:
					y1 = 0
				face = img[y1:y2, x1:x2]
				image = Image.fromarray(face)
				image = image.resize((224,224))
				face_array = np.asarray(image)

				samples = np.asarray(face_array, 'float32')
				samples = np.expand_dims(samples, axis=0)
				encoding = self.encoder_model(torch.Tensor(samples).to(self.device))
				encodings.append(encoding)
				bbox.append((x1, y1, x2-x1, y2-y1))
			return encodings, bbox
		else:
			return None, None

	def get_celeb_prediction(self, img):
		img = imutils.resize(img, height=1080)
		encs, bbox = self.get_encoding(img)
		data = []
		for index, enc in enumerate(encs):
			cv2.rectangle(img, bbox[index], (255,0,0), 2)
			temp_data = {}
			temp_data["face_bbox"] = bbox[index]
			results = self.ann_index.get_nns_by_vector(enc[0], 10, search_k=-1, include_distances=True)
			dist_threshold = 0.9
			celeb_count_dict = self.get_celeb_name_from_id(results, dist_threshold)
			distance = results[1][0]
			if len(celeb_count_dict)!=0 and max(celeb_count_dict.values()) > 3:
				celeb_name = max(celeb_count_dict, key=celeb_count_dict.get)
				cv2.putText(img, celeb_name.upper(), (bbox[index][0]-5, bbox[index][1] - 5), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 1)
				temp_data["celeb_name"] = celeb_name
				temp_data["confidence"] = round(self.face_distance_to_conf(distance),2)
			else:
				temp_data["celeb_name"] = "unknown"
				temp_data["confidence"] = 0.0
			data.append(temp_data)
		return data, img
	
if __name__=="__main__":
	det = CelebRecognition()
	img = cv2.imread("/Users/shobhit2.gupta/Downloads/test_images/bolly3.jpg")
	output = det.get_celeb_prediction(img)
	print(output[0])