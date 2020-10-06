Create your own celeb model
====================================

-  Create a dataset of celebs in the following directory structure:
   A root folder (say ``celeb_images``), inside this should be the folders corresponding to each of the celebs inside which would be the individual pics of the celebs.
-  Each folder name will be considered as the corresponding celeb name
   for the model (WARNING: Do not provide any special characters or
   spaces in the names).
-  Make sure each image has only 1 face (of the desired celebrity), if
   there are multiple faces, only the first detected face will be
   considered.
-  Provide path to the dataset folder (for example, ``celeb_images``
   folder) in the create_celeb_model.py file.
-  Run create_celeb_model.py file.
-  Upon successful completion of the code, we get ``celeb_mapping.json``
   (for storing indexes vs celeb names), ``celeb_index.ann`` (ann file
   for searching encodings) and ``celeb_name_encoding.pkl`` files (for
   storing encodings vs indexes for each celeb). (WARNING: You need to
   provide paths for storing each of these files, default is to store in
   the current directory)