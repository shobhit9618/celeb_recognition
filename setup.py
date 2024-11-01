import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="celeb_detector",
    version="0.0.25",
    author="Shobhit Gupta",
    author_email="shobhit9618@gmail.com",
    description="Model to recognize celebrities using a face matching algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shobhit9618/celeb_recognition",
    packages=setuptools.find_packages(),
    entry_points={
        'console_scripts': [
            'celeb_detector=celeb_detector.celeb_recognition:main'
        ]
    },
    install_requires=[
        'annoy',
        'face_recognition',
        'imutils',
        'opencv-python',
        'matplotlib',
        'numpy',
        'tqdm',
        'pillow',
        'torch',
        'onnx2torch',
        'requests',
        'click'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)