import os
from os.path import expanduser
from PIL import Image
import requests
import cv2
import numpy as np
from IPython.display import display
from celeb_utils.download_gdrive import download_file_from_google_drive
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

home = expanduser("~")
celeb_ann_destination = os.path.join(home,'celeb_index_60.ann')
celeb_mapping_destination = os.path.join(home,'celeb_mapping.json')

celeb_mapping_file_id = '1wDaaSQ6NjxLkxpzYyTRknefizZUKnKDj'
if not os.path.exists(celeb_mapping_destination):
	download_file_from_google_drive(celeb_mapping_file_id, celeb_mapping_destination)

celeb_ann_id = '1-3Wb7fiINbrk9FSagTxjLdSjp7KzrMp7'
if not os.path.exists(celeb_ann_destination):
	download_file_from_google_drive(celeb_ann_id, celeb_ann_destination)

# provide path to image for prediction
url = '' # provide image url here
img = cv2.cvtColor(np.array(Image.open(requests.get(url, stream=True).raw)), cv2.COLOR_BGR2RGB)
# image_path = 'celeb_images/sample_images/sample_image_multi.jpg'
# img = cv2.imread(image_path)

from celeb_utils.celeb_utils import get_celeb_prediction
pred, img = get_celeb_prediction(img)
if pred is not None:
	os.makedirs('celeb_output', exist_ok=True)
	out_im_path = 'celeb_output/image_output.jpg'
	cv2.imwrite(out_im_path, img)
	print("Output image saved at {}".format(out_im_path))

	print("Found celebrities:")
	for c in pred:
		if c["celeb_name"].lower() !="unknown":
			print(c["celeb_name"])

	print("\nOverall output:\n",pred)
else:
	print("No faces detected in the image")