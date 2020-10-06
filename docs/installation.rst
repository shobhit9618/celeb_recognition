Installation
============

-  Run ``pip install -r requirements.py`` to install all the
   dependencies (preferably in a virtual environment).

PyPI package
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


