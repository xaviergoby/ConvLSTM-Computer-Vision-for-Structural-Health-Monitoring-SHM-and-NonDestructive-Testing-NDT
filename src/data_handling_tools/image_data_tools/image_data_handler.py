import os
import settings
import cv2
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from src.utils import file_manipulation_tools
from src.utils import data_preprocessing

class ImageDataHandler:

	def __init__(self, dataset_dir_name):
		self.dataset_dir_name = dataset_dir_name
		self.dataset_dir_path = os.path.join(settings.IMAGES_DIR, self.dataset_dir_name)

		self.training_dir_path = os.path.join(self.dataset_dir_path, "training")
		self.validation_dir_path = os.path.join(self.dataset_dir_path, "validation")
		self.test_dir_path = os.path.join(self.dataset_dir_path, "test")

		self.data_subset_paths_dict = {"training": self.training_dir_path,
		                               "validation": self.validation_dir_path,
		                               "test": self.test_dir_path}

		self.class_labels = list(map(int, file_manipulation_tools.get_file_folder_names_in_dir(self.training_dir_path)))
		self.num_class_labels = len(self.class_labels)

		self.num_training_images_per_label = file_manipulation_tools.get_num_files_in_dir(os.path.join(self.training_dir_path, str(self.class_labels[0])))
		self.num_validation_images_per_label = file_manipulation_tools.get_num_files_in_dir(
			os.path.join(self.validation_dir_path,
			             str(self.class_labels[0])))
		self.num_test_images_per_label = file_manipulation_tools.get_num_files_in_dir(
			os.path.join(self.test_dir_path,
			             str(self.class_labels[0])))
		self.tot_num_training_images = self.num_class_labels * self.num_training_images_per_label
		self.tot_num_validation_images = self.num_class_labels * self.num_validation_images_per_label
		self.tot_num_test_images = self.num_class_labels * self.num_test_images_per_label

	@staticmethod
	def one_got_encode_class_labels(y):
		"""
		This staticmethod is meant for converting/transforming a/your np.ndarray of
		integer encode class labels into a one hot encoded format.
		:param y:
		:return:
		"""
		y_reshaped = y.reshape((y.shape[0], 1))
		one_hot_encoder = OneHotEncoder(sparse=False)
		y_ohe_cls_labels = one_hot_encoder.fit_transform(y_reshaped)
		return y_ohe_cls_labels

	def get_dataset_images_and_labels(self, dataset_split_name, mode="rgb"):
		"""
		:param dataset_split_name: e.g. "training"
		:param mode: Is "rgb" (for frame_channels=0) by def. mode="gray_scale" loads dataset_split_name
		in gray scale format (frame_channels=1).
		:return: np.ndarray with shape of (# of image_datasets, frame_height, frame_width, frame_channels)
		"""
		X_list = []
		y_list = []
		for class_label in self.class_labels:
			label_i_dir_path = os.path.join(self.data_subset_paths_dict[dataset_split_name], str(class_label))
			label_i_dir_file_names = os.listdir(label_i_dir_path)
			for img_i_name in label_i_dir_file_names:
				img_i_path = os.path.join(label_i_dir_path, img_i_name)
				if mode == "rgb":
					img_array = cv2.imread(img_i_path, cv2.COLOR_BGR2RGB)
				elif mode == "gray_scale":
					img_array_brg = cv2.imread(img_i_path)
					img_array = cv2.cvtColor(img_array_brg, cv2.COLOR_BGR2GRAY)
					# print(f"img_array.shape: {img_array.shape}")
					img_array = np.expand_dims(img_array, axis=-1)
					# print(f"img_array.shape: {img_array.shape}")
				else:
					print("Invalid argument passed to parameter mode. Parameter mode may only be 'rgb' or 'gray_scale'!")
					break
				X_list.append(img_array)
				y_list.append(class_label)
		X_array = np.asarray(X_list)
		y_array = np.asarray(y_list)
		y_ohe_cls_labels_array = ImageDataHandler.one_got_encode_class_labels(y_array)
		return X_array, y_ohe_cls_labels_array

	def load_dataset(self, dataset_split_name=None, mode="rgb"):
		"""
		:param dataset_split_name: str of the name of the data split to load. Is None by def and if
		no arg is passed (so None is unchanged) then this function will load each dataset split, i.e.
		the training, validation & testing dataset splits.
		:param mode: Is "rgb" (for frame_channels=0) by def. mode="gray_scale" loads dataset_split_name
		in gray scale format (frame_channels=1).
		:return: a 2-tuple consisting of (dataset_split_inputs, dataset_split_targets) where both
		dataset_split_inputs & dataset_split_targets are np.ndarrays) if the str of the name
		of a particular dataset split is passed as an arg to dataset_split_name, e.g. "training".
		Else, if  dataset_split_name is left unchanged as None (the def opt) then a a 2-tuple
		consisting of (dataset_split_inputs, dataset_split_targets) where both dataset_split_inputs
		& dataset_split_targets are both of type list and of length 3, e.g.
		[training_inputs, validation_inputs, testing_inputs] & [training_targets, validation_targets, testing_targets]
		"""
		input_data = []
		target_data = []
		if dataset_split_name is None:
			for dataset_split_i_name in list(self.data_subset_paths_dict.keys()):
				dataset_split_i = self.get_dataset_images_and_labels(dataset_split_name=dataset_split_i_name, mode=mode)
				dataset_split_i_inputs = dataset_split_i[0]
				dataset_split_i_targets = dataset_split_i[1]
				input_data.append(dataset_split_i_inputs)
				target_data.append(dataset_split_i_targets)
			return input_data, target_data
		elif dataset_split_name is not None:
			dataset_split = self.get_dataset_images_and_labels(dataset_split_name=dataset_split_name, mode=mode)
			return dataset_split[0], dataset_split[1]

	@staticmethod
	def trim_images_width(X, max_img_width):
		untrimmed_X_array = X
		untrimmed_img_width = untrimmed_X_array.shape[2]
		img_left_end_width_init_px_idx = untrimmed_img_width - max_img_width
		trimmed_width_X_array = untrimmed_X_array[:, :, img_left_end_width_init_px_idx:, :]
		return trimmed_width_X_array

	@staticmethod
	def gen_img_frames(X, frame_width):
		"""
		:param X: the np array w/ shape (# of imgs in dataset_name, frame_height, frame_width, frame_channels) to generate frames from
		:param frame_width:
		:return: an array with shape, i.e. (90, 164, 247, 25, 1) where (samples #, frames #, frame_height, frame_width, frame_channels)
		"""
		frames = []
		for X_i in X:
			img_X_i_height = X_i.shape[0]
			img_X_i_width = X_i.shape[1]
			img_X_i_num_frames = int(img_X_i_width / frame_width)
			X_i_frames = []
			lower_bound_px_idx = 0
			for X_i_frame_i_idx in range(img_X_i_num_frames):
				X_i_frame_i = X_i[:, lower_bound_px_idx:lower_bound_px_idx + frame_width, :]
				lower_bound_px_idx = lower_bound_px_idx + frame_width
				X_i_frames.append(X_i_frame_i)
			frames.append(X_i_frames)
		frames_array = np.asarray(frames)
		return frames_array

	def gen_labelled_frames_batches(self, X, y, frame_width):
		untrimmed_images = X
		og_img_width = X[0].shape[1]
		max_img_width = data_preprocessing.get_compatible_img_and_frame_widths(og_img_width, frame_width)
		trimmed_images = self.trim_images_width(untrimmed_images, max_img_width)
		all_X_frames = self.gen_img_frames(trimmed_images, frame_width)
		return all_X_frames, y

	# def get_img_data_frames_and_labels(self, frame_width, dataset_name):
	# 	dataset = self.get_dataset_images_and_labels(dataset_name, mode="rgb")
	# 	X = dataset[0]
	# 	y = dataset[1]
	# 	all_X_frames = self.gen_img_frames(X, frame_width)
	# 	return all_X_frames, y


if __name__ == "__main__":
	from src.utils import data_preprocessing

	data_dir_name = "scenario_1"
	dataset_dir_name = "training"
	# train_dataset_dir_name = "test"
	src = ImageDataHandler(data_dir_name)
	print(f"src.class_labels: {src.class_labels}")
	print(f"src.num_class_labels: {src.num_class_labels}")
	print(f"src.num_training_images: {src.num_training_images_per_label}")
	print(f"src.num_validation_images: {src.num_validation_images_per_label}")
	print(f"src.num_test_images: {src.num_test_images_per_label}")
	img_width = 4101
	frame_width = 25
	max_img_width = data_preprocessing.get_compatible_img_and_frame_widths(img_width, frame_width)
	channels_rgb = "rgb"
	channels_gray = "gray_scale"
	# X, y = src.get_dataset_images_and_labels(dataset_dir_name, channels_rgb)
	input_dataset_splits, target_dataset_splits = src.load_dataset(mode="rgb")
	X = input_dataset_splits[0]
	y = target_dataset_splits[0]
	frames, labels = src.gen_labelled_frames_batches(X, y, frame_width)
	print(f"frames.shape: {frames.shape}")
	print(f"labels.shape: {labels.shape}")
	print(f"frames[0].shape: {frames[0].shape}")
	print(f"labels[0].shape: {labels[0].shape}")
	print(f"len(frames[0]): {len(frames[0])}")
	print(f"len(labels[0]): {len(labels[0])}")
