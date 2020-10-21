import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import configs_and_settings
import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from src.utils import data_preprocessing


# img_path = "data/SPIE2019/All/prototyping_data/Sample2.png
# The factors of 4,100. Answer : 1,2,4,5,10,20,25,41,50,82,100,164,205,410,820,1025,2050,4100,

class ImageDataSource:
	
	def __init__(self, dir_name, train_dir_path=r"data\images\train", validation_dir_path=r"data\images\validation", test_dir_path=r"data\images\test"):
		self.dir_name = dir_name
		self.data_dir_path = os.path.join(configs_and_settings.IMAGES_DIR, dir_name)
		self.train_dir_path = os.path.join(self.data_dir_path, "training")
		self.validation_dir_path = os.path.join(self.data_dir_path, "validation")
		self.test_dir_path = os.path.join(self.data_dir_path, "test")
		# self.validation_dir_path = configs_and_settings.VAL_DIR_PATH
		# self.test_dir_path = configs_and_settings.TEST_DIR_PATH
		# self.test_dir_path = configs_and_settings.TEST_DIR_PATH
		# self.ref_dir = self.test_dir_path
		self.data_subset_paths_dict = {"training":self.train_dir_path, "validation":self.validation_dir_path, "test":self.test_dir_path}
		# self.data_dir_paths_dict = {"train": self.train_dir_path, "validation": self.validation_dir_path,
		# 							"test": self.test_dir_path}
		self.class_labels = [int(label) for label in os.listdir(self.test_dir_path)]
	
	def get_dataset_img_dir_path_by_label(self, label, dataset_dir="training"):
		given_label_dataset_dir_path = os.path.join(self.data_subset_paths_dict[dataset_dir], str(label))
		return given_label_dataset_dir_path
	
	def get_dataset_img_file_names_by_label(self, label, dataset_dir="training"):
		given_label_dataset_img_file_names = os.listdir(self.get_dataset_img_dir_path_by_label(label, dataset_dir))
		return given_label_dataset_img_file_names
	
	def get_dataset_img_file_paths_by_label(self, label, dataset_dir="training"):
		img_file_paths = [os.path.join(self.get_dataset_img_dir_path_by_label(label, dataset_dir), img_file_name) for img_file_name in
		                  self.get_dataset_img_file_names_by_label(label, dataset_dir)]
		return img_file_paths
	
	def get_dataset_img_file_paths_dict(self, dataset_dir="training"):
		dataset_img_file_paths_dict = {}
		class_labels = self.class_labels
		for class_label in class_labels:
			dataset_img_file_path = self.get_dataset_img_file_paths_by_label(class_label, dataset_dir)
			dataset_img_file_paths_dict[class_label] = dataset_img_file_path
		return dataset_img_file_paths_dict
	
	def get_dataset_img_arrays_and_labels_dict(self, dataset_dir="training"):
		img_arrays_dict = {}
		dataset_img_file_paths_dict = self.get_dataset_img_file_paths_dict(dataset_dir)
		class_labels = self.class_labels
		for class_label in class_labels:
			label_i_img_file_paths = dataset_img_file_paths_dict[class_label]
			label_i_img_arrays = []
			for img_file_path in label_i_img_file_paths:
				img_array = cv2.imread(img_file_path, 0)
				label_i_img_arrays.append(img_array)
			img_arrays_dict[class_label] = label_i_img_arrays
		
		return img_arrays_dict
	
	def get_right_end_trimmed_imgs_dataset(self, max_img_width=configs_and_settings.MAX_VALID_IMG_WIDTH, dataset_name="training"):
		"""
		:param max_img_width: the max allowable frame_width of each img, should obv be less than or = to  the original frame_width of an image(es).
		Is configs_and_settings.MAX_VALID_IMG_WIDTH (=4100) by def.
		:param dataset_name: name of the subset of image_datasets data, i.e. "training", "validation" or "test". Is "test" by def
		:return:
		"""
		print(f"dataset_name: {dataset_name}")
		data_set_path = self.data_subset_paths_dict[dataset_name.lower()]
		# print(f"data_set_path: {data_set_path}")
		X = []
		y = []
		for class_label in self.class_labels:
			data_set_label_i_dir_path = os.path.join(data_set_path, str(class_label))
			# print(f"data_set_label_i_dir_path: {data_set_label_i_dir_path}")
			data_set_label_i_image_file_names = os.listdir(data_set_label_i_dir_path)
			for image_i in data_set_label_i_image_file_names:
				image_i_file_path = os.path.join(data_set_label_i_dir_path, image_i)
				img_array = cv2.imread(image_i_file_path, 0)
				img_left_end_width_init_px_idx = img_array.shape[1] - max_img_width
				# print(f"img_array.shape: {img_array.shape}")
				img_array = img_array[:, img_left_end_width_init_px_idx:]  ###########
				# print(f"img_array.shape: {img_array.shape}")
				X.append(img_array)
				y.append(class_label)
		X = np.asarray(X)
		X = np.expand_dims(X, axis=-1)
		X = X.reshape(X.shape[0], X.shape[1], X.shape[2], X.shape[-1])
		y = np.asarray(y)
		y = self.one_got_encode_class_labels(y)
		# print(f"X_train.shape: {X_train.shape}")
		# print(f"y_train.shape: {y_train.shape}")
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
		:return: an array with shape, i.e. (90, 164, 247, 25, 1) where (samples #, frames #, frame_height, frame_width, frame_channels)
		"""
		# e.g. frame_width = 1367 yields 3 frames!
		# img_width = X_train.shape[2]
		# frames_cnt = int(img_width / frame_width)
		# print(f"X_train[0].shape: {X_train[0].shape}")
		# print(f"X_train.shape: {X_train.shape}")
		# print(f"X_train.shape[3]: {X_train.shape[3]}")
		# print(f"X_train.shape[2]: {X_train.shape[2]}")
		# print(f"frames_cnt: {frames_cnt}")
		# print(frames_cnt)
		frames = []
		for X_i in X:
			img_X_i_height = X_i.shape[0]
			img_X_i_width = X_i.shape[1]
			# print(f"X_i.shape: {X_i.shape}")
			img_X_i_num_frames = int(img_X_i_width / frame_width)
			X_i_frames = []
			lower_bound_px_idx = 0
			for X_i_frame_i_idx in range(img_X_i_num_frames):
				X_i_frame_i = X_i[:, lower_bound_px_idx:lower_bound_px_idx + frame_width, :]
				lower_bound_px_idx = lower_bound_px_idx + 25
				X_i_frames.append(X_i_frame_i)
			frames.append(X_i_frames)
		frames_array = np.asarray(frames)
		return frames_array
	
	def get_img_data_frames_and_labels(self, frame_width=configs_and_settings.FRAME_WIDTH, max_img_width=configs_and_settings.MAX_VALID_IMG_WIDTH, dataset_name="training"):
		dataset = self.get_right_end_trimmed_imgs_dataset(max_img_width, dataset_name)
		X = dataset[0]
		y = dataset[1]
		all_X_frames = self.gen_img_frames(X, frame_width)
		return all_X_frames, y


if __name__ == "__main__":
	frame_width = 25
	src = ImageDataSource("scenario_1")
	# dataset = "training"
	dataset = "validation"
	# dataset = "test"
	data = src.get_right_end_trimmed_imgs_dataset(dataset_name=dataset)
	X = data[0]
	y = data[1]
	print(f"dataset: {dataset}")
	print(f"X_train.shape: {X.shape}")
	print(f"y.shape: {y.shape}")
	max_img_width = data_preprocessing.get_compatible_img_and_frame_widths(4101, 25)
	frames_and_labels = src.get_img_data_frames_and_labels(max_img_width=max_img_width, dataset_name=dataset)
	frames = frames_and_labels[0]
	labels = frames_and_labels[1]
	print(f"frames.shape: {frames.shape}")
	print(f"labels.shape: {labels.shape}")
	print(f"frames[0].shape: {frames[0].shape}")
	print(f"labels[0].shape: {labels[0].shape}")
	print(f"len(frames[0]): {len(frames[0])}")
	print(f"len(labels[0]): {len(labels[0])}")
	
