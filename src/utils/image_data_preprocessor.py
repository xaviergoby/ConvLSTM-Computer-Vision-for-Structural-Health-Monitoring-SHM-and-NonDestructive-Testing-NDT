import numpy as np
import cv2
from sklearn.preprocessing import OneHotEncoder



class ImageDataSetPreProcessor:
	
	def __init__(self, image_dataset):
		self.image_dataset = image_dataset
	
	
	def get_compatible_img_and_frame_widths(self, img_width, frame_width):
		"""
		This function is meant for finding the number closest to n and divisible by m. Within the context
		of data postprocessing of image_datasets, this function is used for determining the maximum acceptable frame_width
		of an image(es) (img_width) for creating frames for the image(es) of constant frame_width (frame_width).
		gcd: greatest common divisor
		:param img_width: pixel frame_width (AKA milliseconds) of an image
		:param frame_width: pixel frame_width (AKA milliseconds) of a frame
		:return:
		"""
		q = int(img_width / frame_width)
		n1 = frame_width * q
		if ((img_width * frame_width) > 0):
			n2 = (frame_width * (q + 1))
		else:
			n2 = (frame_width * (q - 1))
		if (abs(img_width - n1) < abs(img_width - n2)):
			return n1
		return n2
	
	def normalize_img(self, img):
		if len(img.shape) == 3:
			if img.shape[-1] != 1:
				# if img.shape[-1] != 1:
				img_bgr = img
				img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
		img_place_holder = np.zeros(img.shape)
		img_normed = cv2.normalize(img, img_place_holder, 0, 255, cv2.NORM_MINMAX)
		return img_normed
	
	def ohe_class_labels(self, y):
		"""
		This method is meant for converting/transforming a/your np.ndarray of
		integer encode class labels into a one hot encoded format.
		:param y: array type of outputs
		:return: 2D array type of one-hot encoding transformation result
		"""
		y_reshaped = y.reshape((y.shape[0], 1))
		one_hot_encoder = OneHotEncoder(sparse=False)
		y_ohe_cls_labels = one_hot_encoder.fit_transform(y_reshaped)
		return y_ohe_cls_labels
	
	def convert_rgb_img_to_grey_scale(self, rgb_img):
		grey_scale_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
		return grey_scale_img
	
	def convert_rgb_images_dataset_to_grey_scale(self, rgb_images_array):
		grey_scale_images_list = []
		for rgb_img_i in range(len(rgb_images_array.shape[0])):
			grey_scale_img = self.convert_rgb_img_to_grey_scale(rgb_images_array[rgb_img_i])
			grey_scale_images_list.append(grey_scale_img)
		grey_scale_images_array = np.array(grey_scale_images_list)
		return grey_scale_images_array