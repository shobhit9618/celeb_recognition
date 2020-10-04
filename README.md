# Celebrity Recognition
Model to recognize celebrities using a face matching algorithm.

Model is based on a dataset of around 6000 images of 60 celebrities (100 each).

## Basic working of the algorithm includes the following:
- Face detection is done using MTCNN face detection model.

- Face encodings are created using [VGGFace](https://github.com/rcmalli/keras-vggface) model in keras.

- Face matching is done using [annoy](https://github.com/spotify/annoy) library (spotify).

## Installing dependencies
- Run `pip install -r requirements.py` to install all the dependencies (preferably in a virtual environment).

## PyPI package
### Installation
- To ensure you have all the required additional packages, run `pip install -r requirements.py` first.
- To install pip package, run:
    ```bash
    # pip release version (on test PyPI only as of now)
    pip3 install --user -i https://test.pypi.org/simple/ celeb-detector==0.0.14
    # Directly from repo
    pip3 install git+https://github.com/shobhit9618/celeb_recognition.git
    ```

### Using pip pakcage
- For using my model for predictions, use the following lines of code after installation:
    ```python
    import celeb_detector
    img_path = 'sample_image.jpg'
    celeb_detector.celeb_recognition(img_path)
    ```

- For using you own custom model, also provide path to json and ann files as shown below:
    ```python
    import celeb_detector
    img_path = 'sample_image.jpg'
    ann_path = 'sample_index.ann'
    celeb_map = 'sample_mapping.json'
    celeb_detector.celeb_recognition(img_path, ann_path, celeb_map)
    ```

- For creating your own model, use as follows:
    ```python
    import celeb_detector
    folder_path = 'sample_folder'
    celeb_detector.create_celeb_model(folder_path)
    ```

- NOTE: pip package is unstable as of now, it is recommended to use python files from the repo for creating your model and making predictions. Details for the same are provided below.

## Create your own celeb model
- Create a dataset of celebs in the following directory structure:
    ```bash
    celeb_images/
        celeb-a/
            celeb-a_1.jpg
            celeb-a_2.jpg
            ...
        celeb-b/
            celeb-b_1.jpg
            celeb-b_1.jpg
            ...
        ...
    ```
- Each folder name will be considered as the corresponding celeb name for the model (WARNING: Do not provide any special characters or spaces in the names).
- Make sure each image has only 1 face (of the desired celebrity), if there are multiple faces, only the first detected face will be considered.
- Provide path to the dataset folder (for example, `celeb_images` folder) in the [create_celeb_model.py](create_celeb_model.py) file.
- Run [create_celeb_model.py](create_celeb_model.py) file.
- Upon successful completion of the code, we get `celeb_mapping.json` (for storing indexes vs celeb names), `celeb_index.ann` (ann file for searching encodings) and `celeb_name_encoding.pkl` files (for storing encodings vs indexes for each celeb).
(WARNING: You need to provide paths for storing each of these files, default is to store in the current directory)

## Model predictions in jupyter
- Provide paths to `celeb_mapping.json` and `celeb_index.ann` files in [celeb_recognition.ipynb](celeb_recognition.ipynb) file. If you want to try my model, ignore this step.
- Run all the cells in the [celeb_recognition.ipynb](celeb_recognition.ipynb) file, the final cell will provide widgets for uploading images and making predictions
(this will also download the necessary model files).
- NOTE: [celeb_recognition.ipynb](celeb_recognition.ipynb) is a standalone file and does not require any other files from the repo for running.

## Model predictions in python
- Provide paths to `celeb_mapping.json` and `celeb_index.ann` files in [celeb_recognition.py](celeb_recognition.py) and [celeb_utils.py](celeb_utils/celeb_utils.py) files. If you want to try my model, ignore this step.
- Run [celeb_recognition.py](celeb_recognition.py) file, provide path to image in the file.
- Output includes a list of the identified faces, bounding boxes and the predicted celeb name (unknown if not found).
- It also displays the output with bounding boxes.

## Sample image output
![Image](celeb_utils/sample_images/sample_output_multi.png)

## Binder
You can run a binder application by clicking the following link:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/shobhit9618/celeb_recognition/main)

You can also launch a voila binder application (which only has widgets for image upload and celeb prediction) by clicking [here](https://mybinder.org/v2/gh/shobhit9618/celeb_recognition/main?urlpath=%2Fvoila%2Frender%2Fceleb_recognition.ipynb).

## Google Colab
To open and run [celeb_recognition.ipynb](celeb_recognition.ipynb) file in google colab, click the following link:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shobhit9618/celeb_recognition/blob/main/celeb_recognition.ipynb)
