# Celebrity Recognition [![PyPI version](https://badge.fury.io/py/celeb-detector.svg)](https://badge.fury.io/py/celeb-detector) [![Documentation Status](https://readthedocs.org/projects/celeb-recognition/badge/?version=main)](https://celeb-recognition.readthedocs.io/en/main/) [![Anaconda-Server Badge](https://anaconda.org/shobhit9618/celeb-detector/badges/installer/env.svg)](https://anaconda.org/shobhit9618/celeb-detector)
Model to recognize celebrities using a face matching algorithm.

Model is based on a dataset of around 6000 images of 60 celebrities (100 each).

Refer [this](https://celeb-recognition.readthedocs.io/en/main/) for detailed documentation.

## Basic working of the algorithm includes the following:
- Face detection is done using MTCNN face detection model.

- Face encodings are created using [VGGFace](https://github.com/rcmalli/keras-vggface) model in keras.

- Face matching is done using [annoy](https://github.com/spotify/annoy) library (spotify).