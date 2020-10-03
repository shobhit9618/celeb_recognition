import os
import requests

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

celeb_mapping_file_id = '1wDaaSQ6NjxLkxpzYyTRknefizZUKnKDj'
celeb_mapping_destination = 'celeb_mapping.json'
if not os.path.exists(celeb_mapping_destination):
    download_file_from_google_drive(celeb_mapping_file_id, celeb_mapping_destination)

celeb_ann_id = '1-3Wb7fiINbrk9FSagTxjLdSjp7KzrMp7'
celeb_ann_destination = 'celeb_index_60.ann'
if not os.path.exists(celeb_ann_destination):
    download_file_from_google_drive(celeb_ann_id, celeb_ann_destination)

path0 = "~/.keras"
os.makedirs(path0, exist_ok=True)
path1 = "~/.keras/models"
os.makedirs(path1, exist_ok=True)
path2 = "~/.keras/models/vggface"
os.makedirs(path2, exist_ok=True)

vggface_model_path = ".keras/models/vggface/rcmalli_vggface_tf_notop_resnet50.h5"
if not os.path.exists(vggface_model_path):
    os.system("wget https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_notop_resnet50.h5 -O {}".format(vggface_model_path))