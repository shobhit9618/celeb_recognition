import torch

from celeb_detector.celeb_recognition import celeb_recognition

def find_celeb(image_path):
	return celeb_recognition(image_path)