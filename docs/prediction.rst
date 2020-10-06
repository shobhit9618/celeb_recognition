Predictions
===========

Model predictions in jupyter
------------------------------

-  Provide paths to ``celeb_mapping.json`` and ``celeb_index.ann`` files
   in celeb_recognition.ipynb file. If
   you want to try my model, ignore this step.
-  Run all the cells in the celeb_recognition.ipynb file, the
   final cell will provide widgets for uploading images and making
   predictions (this will also download the necessary model files).
-  NOTE: celeb_recognition.ipynb is a standalone file and does not 
   require any other files from the repo
   for running.

Model predictions in python
------------------------------

-  Provide paths to ``celeb_mapping.json`` and ``celeb_index.ann`` files
   in celeb_recognition.py and celeb_utils/celeb_utils.py files. If you 
   want to try my model, ignore this step.
-  Run celeb_recognition.py file, provide
   path to image in the file.
-  Output includes a list of the identified faces, bounding boxes and
   the predicted celeb name (unknown if not found).
-  It also displays the output with bounding boxes.

Sample image output
-------------------

.. figure:: https://drive.google.com/uc?export=view&id=1W4P0PPLjr0BHDkj2CzLgFGpOYn4MF1Ck
   :alt: Image

   Image