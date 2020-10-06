Installing dependencies
====================================

-  Run ``pip install -r requirements.py`` to install all the
   dependencies (preferably in a virtual environment).

PyPI package
====================================

Installation
------------

- To ensure you have all the required additional packages, run ``pip install -r requirements.py`` first.
- To install pip package, run::

   	# pip release version    
   	pip install celeb-detector   
   	# also install additional dependencies with this (if not installed via requirements.txt file)     
   	pip install annoy keras-vggface keras-applications   
   	# Directly from repo     
   	pip install git+https://github.com/shobhit9618/celeb_recognition.git

- If you are using conda on linux or ubuntu, you can use the following commands to create and use a new environment called celeb-detector::

	conda env create shobhit9618/celeb-detector
	conda activate celeb-detector

This will install all the required dependencies. To ensure you are using the latest version of the package, also run (inside the environment)::

	pip install --upgrade celeb-detector

Using pip pakcage
-----------------

-  For using my model for predictions, use the following lines of code
   after installation::
   
      import celeb_detector # on running for the first time, this will 
      download vggface model     
      img_path = 'sample_image.jpg'     
      celeb_detector.celeb_recognition(img_path) # on running for the first time, 2 files (celeb_mapping.json and celeb_index_60.ann) will be downloaded to the home directory
      
      # if you want to use an image url, just provide the url and add url=True
      url = 'https://sample/sample_image_url.jpg'
      celeb_detector.celeb_recognition(url, url=True)
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
