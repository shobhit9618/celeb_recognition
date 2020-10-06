Celebrity Recognition
=====================

Model to recognize celebrities using a face matching algorithm.

Model is based on a dataset of around 6000 images of 60 celebrities (100
each).

Basic working of the algorithm
------------------------------

-  Face detection is done using MTCNN face detection model.

-  Face encodings are created using
   `VGGFace <https://github.com/rcmalli/keras-vggface>`__ model in
   keras.

-  Face matching is done using
   `annoy <https://github.com/spotify/annoy>`__ library (spotify).
