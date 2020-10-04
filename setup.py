import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="celeb_detector", # Replace with your own username
    version="0.0.1",
    author="Shobhit Gupta",
    author_email="shobhit9618@gmail.com",
    description="Model to recognize celebrities using a face matching algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shobhit9618/celeb_recognition",
    packages=setuptools.find_packages(),
    install_requires = ['mtcnn',
                        'tensorflow',
                        'keras>=2.4.3',
                        'keras-vggface>=0.5',
                        'imutils',
                        'opencv-python>=4.0',
                        'matplotlib',
                        'numpy'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)