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


	def load_dataset(self, data_set_name="test"):
		"""

		:param data_set_name:
		:return: X w/ shape (
		"""
		data_set_path = self.data_dir_paths_dict[data_set_name.lower()]
		image_data_file_paths_dict = {}
		X = []
		y = []
		for class_label in self.class_labels:
			data_set_label_i_dir_path = os.path.join(data_set_path, str(class_label))
			data_set_label_i_image_file_names = os.listdir(data_set_label_i_dir_path)
			for image_i in data_set_label_i_image_file_names:
				image_i_file_path = os.path.join(data_set_label_i_dir_path, image_i)
				img_array = cv2.imread(image_i_file_path, 0)
				X.append(img_array)
				y.append(class_label)
		X = np.asarray(X)
		X = np.expand_dims(X, axis=-1)
		# X = np.reshape(X, (X.shape[0], X.shape[2], X.shape[1], X.shape[-1]))
		X = X.reshape(X.shape[0], X.shape[2], X.shape[1], X.shape[-1])
		y = np.asarray(y)
		y = self.one_got_encode_class_labels(y)
		return X, y


	def one_got_encode_class_labels(self, y):
		y = y.reshape((y.shape[0], 1))
		onehot_encoder = OneHotEncoder(sparse=False)
		y = onehot_encoder.fit_transform(y)
		return y


	def generate_X_data_per_imaage_frames(self, X, frame_width=1367, num_frames=None):
		# e.g. frame_width = 1367 yields 3 frames!
		frames_cnt = int(X.shape[1]/frame_width)
		frames = []
		for X_i in X:
			frame_i = 1
			X_i_frames = []
			for frame_cnt_i in range(1, frames_cnt+1):
				X_i_frame_i = X_i[:frame_width*frame_cnt_i,:,:]
				X_i_frames.append(X_i_frame_i)
				# frame_i = frame_i + 1
			frames.append(X_i_frames)
		frames_array = np.asarray(frames)
		return frames_array
		pass






if __name__ == "__main__":
	# img_file_names = LSTMDataSource().get_image_file_names_by_label(0)
	# x = ImageDataSource().get_test_image_file_paths_by_label(0)
	# x = os.path.join(os.path.dirname(os.getcwd()), r"data\images\test")
	src = ImageDataSource()
	# src.get_test_imageE_file_paths_dict()
	res = src.load_dataset()
	X = res[0]
	y = res[1]
	frames = src.generate_X_data_per_imaage_frames(X)
	print(f"X.shape: {X.shape}")
	print(f"y.shape: {y.shape}")
	print(f"frames.shape: {frames.shape}")
	# l = src.get_test_image_file_paths_dict()
	# print(len(l))
	# data = src.get_test_img_arrays_and_labels_dict()
	# print(data)
	# print(len(data))