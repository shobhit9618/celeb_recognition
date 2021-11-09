import os
from os.path import expanduser
import requests
import cv2
from PIL import Image
from celeb_detector.download_gdrive import download_file_from_google_drive
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
import click

home = expanduser("~")
celeb_ann_destination = os.path.join(home,'celeb_index_60.ann')
celeb_mapping_destination = os.path.join(home,'celeb_mapping.json')
# provide path to image for prediction
def celeb_recognition(image_path, ann_filepath=None, celeb_mapping_path = None, save_img_output=False):
    if celeb_mapping_path is None:
        celeb_mapping_path = celeb_mapping_destination
        celeb_mapping_file_id = '1wDaaSQ6NjxLkxpzYyTRknefizZUKnKDj'
        if not os.path.exists(celeb_mapping_destination):
            download_file_from_google_drive(celeb_mapping_file_id, celeb_mapping_destination)

    if ann_filepath is None:
        ann_filepath = celeb_ann_destination
        celeb_ann_id = '1-3Wb7fiINbrk9FSagTxjLdSjp7KzrMp7'
        if not os.path.exists(celeb_ann_destination):
            download_file_from_google_drive(celeb_ann_id, celeb_ann_destination)
    # image_path = 'celeb_images/sample_images/sample_image_multi.jpg'
    try:
        img = cv2.cvtColor(np.array(Image.open(requests.get(image_path, stream=True).raw)), cv2.COLOR_BGR2RGB)
    except Exception as e:
        if not os.path.exists(image_path):
            raise FileNotFoundError("Invalid path: {0}".format(image_path))
        img = cv2.imread(image_path)

    from celeb_detector.celeb_utils import get_celeb_prediction
    pred, img_out = get_celeb_prediction(img, ann_filepath, celeb_mapping_path)
    if pred is not None:
        if save_img_output:
            os.makedirs('celeb_output', exist_ok=True)
            out_im_path = 'celeb_output/image_output.jpg'
            cv2.imwrite(out_im_path, img_out)
            print("Output image saved at {}".format(out_im_path))

        print("Found celebrities:")
        for c in pred:
            if c["celeb_name"].lower() !="unknown":
                print(c["celeb_name"])

        print("\nOverall output:\n",pred)
        return pred

    else:
        print("No faces detected in the image")
        return None


@click.command()
@click.option('--image_path', required=True, help='Path/url of image you want to check')
@click.option('--ann_filepath', default=None, help='Path to ann file, if using custom model, else ignore')
@click.option('--celeb_mapping_path', default=None, help='Path to celeb mapper file, if using custom file, else ignore')
@click.option('--url', default=True, type=bool, help='Set to false if you want to detect celeb on local image, default true')

def main(image_path, ann_filepath, celeb_mapping_path, url):
    celeb_recognition(image_path, ann_filepath, celeb_mapping_path, False, url)


if __name__ == "__main__":
    main()


