import os
from PIL import Image
import cv2
from IPython.display import display
from celeb_utils.download_gdrive import download_file_from_google_drive
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

celeb_mapping_file_id = '1wDaaSQ6NjxLkxpzYyTRknefizZUKnKDj'
celeb_mapping_destination = 'celeb_mapping.json'
if not os.path.exists(celeb_mapping_destination):
	download_file_from_google_drive(celeb_mapping_file_id, celeb_mapping_destination)

celeb_ann_id = '1-3Wb7fiINbrk9FSagTxjLdSjp7KzrMp7'
celeb_ann_destination = 'celeb_index_60.ann'
if not os.path.exists(celeb_ann_destination):
	download_file_from_google_drive(celeb_ann_id, celeb_ann_destination)

# provide path to image for prediction
image_path = 'celeb_images/sample_images/sample_image_multi.jpg'
img = cv2.imread(image_path)

from celeb_utils.celeb_utils import get_celeb_prediction
pred, img = get_celeb_prediction(img)
out_im_path = 'celeb_images/sample_images/sample_image_output.jpg'
cv2.imwrite(out_im_path, img)
print("Output image saved at {}".format(out_im_path))

print("Found celebrities:")
for c in pred:
	if c["celeb_name"].lower() !="unknown":
		print(c["celeb_name"])

print("\nOverall output:\n",pred)