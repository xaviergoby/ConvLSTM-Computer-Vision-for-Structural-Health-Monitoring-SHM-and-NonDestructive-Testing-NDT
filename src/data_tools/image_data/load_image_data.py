import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import settings
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# img_path = "data/SPIE2019/All/test/Sample2.png
# The factors of 4,100. Answer : 1,2,4,5,10,20,25,41,50,82,100,164,205,410,820,1025,2050,4100,

class ImageDataSource:

	def __init__(self, train_dir_path = r"data\images\train",
				 validation_dir_path = r"data\images\validation",
				 test_dir_path = r"data\images\test"):
		self.train_dir_path = settings.TRAIN_DIR_PATH
		self.validation_dir_path = settings.VAL_DIR_PATH
		self.test_dir_path = settings.TEST_DIR_PATH
		self.ref_dir = self.test_dir_path
		self.data_dir_paths_dict = {"train":self.train_dir_path, "validation":self.validation_dir_path,
		                            "test":self.test_dir_path}
		self.class_labels = [int(label) for label in os.listdir(self.test_dir_path)]

	def get_test_image_dir_path_by_label(self, label):
		label_test_images_dir_path = os.path.join(self.test_dir_path, str(label))
		return label_test_images_dir_path

	def get_test_image_file_names_by_label(self, label):
		label_test_image_file_names = os.listdir(self.get_test_image_dir_path_by_label(label))
		return label_test_image_file_names

	def get_test_image_file_paths_by_label(self, label):
		img_file_paths = [os.path.join(self.get_test_image_dir_path_by_label(label), img_file_name) for img_file_name in self.get_test_image_file_names_by_label(label)]
		return img_file_paths

	def get_test_image_file_paths_dict(self):
		test_img_file_paths_dict = {}
		class_labels = self.class_labels
		for class_label in class_labels:
			test_imgimg_file_paths = self.get_test_image_file_paths_by_label(class_label)
			test_img_file_paths_dict[class_label] = test_imgimg_file_paths
		return test_img_file_paths_dict

	def get_test_img_arrays_and_labels_dict(self):
		img_arrays_dict = {}
		test_img_file_paths_dict = self.get_test_image_file_paths_dict()
		class_labels = self.class_labels
		for class_label in class_labels:
			label_i_img_file_paths = test_img_file_paths_dict[class_label]
			label_i_img_arrays = []
			for img_file_path in label_i_img_file_paths:
				img_array = cv2.imread(img_file_path, 0)
				label_i_img_arrays.append(img_array)
			img_arrays_dict[class_label] = label_i_img_arrays

		return img_arrays_dict

	def get_dataset(self, data_set_name="test"):
		"""
		:param data_set_name:
		:return: X w/ shape (
		"""
		data_set_path = self.data_dir_paths_dict[data_set_name.lower()]
		X = []
		y = []
		for class_label in self.class_labels:
			data_set_label_i_dir_path = os.path.join(data_set_path, str(class_label))
			data_set_label_i_image_file_names = os.listdir(data_set_label_i_dir_path)
			for image_i in data_set_label_i_image_file_names:
				image_i_file_path = os.path.join(data_set_label_i_dir_path, image_i)
				img_array = cv2.imread(image_i_file_path, 0)
				img_array = img_array[:, 1:] ###########
				X.append(img_array)
				y.append(class_label)
		X = np.asarray(X)
		X = np.expand_dims(X, axis=-1)
		X = X.reshape(X.shape[0], X.shape[1], X.shape[2], X.shape[-1])
		y = np.asarray(y)
		y = self.one_got_encode_class_labels(y)
		return X, y

	def one_got_encode_class_labels(self, y):
		y = y.reshape((y.shape[0], 1))
		onehot_encoder = OneHotEncoder(sparse=False)
		y = onehot_encoder.fit_transform(y)
		return y

	def gen_img_frames(self, X, frame_width=25):
		"""
		:param X:
		:param frame_width:
		:return: an array with shape, i.e. (90, 164, 247, 25, 1) where (samples #, frames #, height, width, channels)
		"""
		# e.g. frame_width = 1367 yields 3 frames!
		frames_cnt = int(X.shape[2]/frame_width)
		print(frames_cnt)
		frames = []
		for X_i in X:
			X_i_frames = []
			lower_bound_px_idx = 0
			for X_i_frame_i_idx in range(frames_cnt):
				X_i_frame_i = X_i[:,lower_bound_px_idx:lower_bound_px_idx+frame_width,:]
				lower_bound_px_idx = lower_bound_px_idx + 25
				X_i_frames.append(X_i_frame_i)
			frames.append(X_i_frames)
		frames_array = np.asarray(frames)
		return frames_array

	def get_img_data_frames_and_labels(self, frame_width=25, data_set_name="test"):
		dataset = self.get_dataset(data_set_name)
		X = dataset[0]
		y = dataset[1]
		all_X_frames = self.gen_img_frames(X, frame_width)
		return all_X_frames, y

if __name__ == "__main__":
	src = ImageDataSource()
	frames_and_labels = src.get_img_data_frames_and_labels()
	frames = frames_and_labels[0]
	labels = frames_and_labels[1]
	print(f"frames.shape: {frames.shape}")
	print(f"labels.shape: {labels.shape}")
	print(f"frames[0].shape: {frames[0].shape}")
	print(f"labels[0].shape: {labels[0].shape}")
	print(f"len(frames[0]): {len(frames[0])}")
	print(f"len(labels[0]): {len(labels[0])}")