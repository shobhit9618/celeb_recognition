Using pip pakcage
=================

-  For using my model for predictions, use the following lines of code
   after installation::
   
      import celeb_detector # on running for the first time, this will 
      download vggface model     
      img_path = 'sample_image.jpg' # this supports both local path and web url like https://sample/sample_image_url.jpg    
      celeb_detector.celeb_recognition(img_path) # on running for the first time, 2 files (celeb_mapping.json and celeb_index_60.ann) will be downloaded to the home directory
      
   This returns a list of dictionaries, each dictionary contains bbox
   coordinates, celeb name and confidence for each face detected in the
   image (celeb name will be unknown if no matching face detected).

-  For using your own custom model, also provide path to json and ann
   files as shown below::    
      import celeb_detector     
      img_path = 'sample_image.jpg'     
      ann_path = 'sample_index.ann'     
      celeb_map = 'sample_mapping.json'     
      celeb_detector.celeb_recognition(img_path, ann_path, celeb_map)

-  For creating your own model (refer next section for more details on usage)
   and run as follows::
      import celeb_detector     
      folder_path = 'celeb_images'     
      celeb_detector.create_celeb_model(folder_path)