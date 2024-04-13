import os
import requests
import cv2
from PIL import Image
import numpy as np
import click
from celeb_detector.celeb_prediction_main import CelebRecognition

# provide path to image for prediction
def get_celebrity(image_path, save_img_output=False):
    celeb_recog = CelebRecognition()
    try:
        img = cv2.cvtColor(np.array(Image.open(requests.get(image_path, stream=True).raw)), cv2.COLOR_BGR2RGB)
    except Exception as e:
        if not os.path.exists(image_path):
            raise FileNotFoundError("Invalid path: {0}".format(image_path))
        img = cv2.imread(image_path)

    pred, img_out = celeb_recog.get_celeb_prediction(img)
    if pred is not None:
        if save_img_output:
            os.makedirs('celeb_output', exist_ok=True)
            out_im_path = 'celeb_output/image_output.jpg'
            cv2.imwrite(out_im_path, img_out)
            print("Output image saved at {}".format(out_im_path))

        return pred

    else:
        return "No faces detected in the image"


@click.command()
@click.option('--image-path', required=True, help='Path/url of image')
@click.option('--save-output', is_flag=True, show_default=True, default=False, help='Save image output')

def main(image_path, save_output=False):
    out = get_celebrity(image_path, save_output)
    print(out)
    return out

if __name__ == "__main__":
    main()


