import os
from PIL import Image
import cv2
from IPython.display import display
from celeb_utils.download_gdrive import download_file_from_google_drive
from celeb_utils.celeb_utils import get_celeb_prediction

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

pred, img = get_celeb_prediction(img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
im_pil = Image.fromarray(img)
display(im_pil)
print(pred)