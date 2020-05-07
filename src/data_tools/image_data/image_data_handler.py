import os
import settings
import cv2
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from src.utils import file_manipulation_tools
from src.utils import data_preprocessing


class ImageDataHandler:
	
	def __init__(self, dir_name):
		self.dir_name = dir_name
		self.data_dir_path = os.path.join(settings.IMAGES_DIR, self.dir_name)
		self.training_dir_path = os.path.join(self.data_dir_path, "training")
		self.validation_dir_path = os.path.join(self.data_dir_path, "validation")
		self.test_dir_path = os.path.join(self.data_dir_path, "test")
		self.data_subset_paths_dict = {"training": self.training_dir_path,
									   "validation": self.validation_dir_path,
									   "test": self.test_dir_path}
		self.class_labels = list(map(int, file_manipulation_tools.get_file_folder_names_in_dir(self.training_dir_path)))
		self.num_class_labels = len(self.class_labels)
		self.num_training_images_per_label = file_manipulation_tools.get_num_files_folders_in_dir(os.path.join(self.training_dir_path,
																											   str(self.class_labels[0])))
		self.num_validation_images_per_label = file_manipulation_tools.get_num_files_folders_in_dir(os.path.join(self.validation_dir_path,
																												 str(self.class_labels[0])))
		self.num_test_images_per_label = file_manipulation_tools.get_num_files_folders_in_dir(os.path.join(self.test_dir_path,
																										   str(self.class_labels[0])))
		self.tot_num_training_images = self.num_class_labels * self.num_training_images_per_label
		self.tot_num_validation_images = self.num_class_labels * self.num_validation_images_per_label
		self.tot_num_test_images = self.num_class_labels * self.num_test_images_per_label


	
	def get_dataset_file_names_by_label(self, label, dataset_name):
		given_label_dataset_img_file_names = os.listdir(os.path.join(self.data_subset_paths_dict[dataset_name], str(label)))
		return given_label_dataset_img_file_names

	def get_dataset_file_paths_by_label(self, label, dataset_name):
		labelled_dataset_file_names = self.get_dataset_file_names_by_label(label, dataset_name)
		labelled_dataset_dir_path = os.path.join(self.data_subset_paths_dict[dataset_name], str(label))
		img_file_paths = [os.path.join(labelled_dataset_dir_path, img_file_name) for img_file_name in
						  labelled_dataset_file_names]
		return img_file_paths

	def get_dataset_label_file_paths_dict(self, dataset_name):
		dataset_file_paths_dict = {}
		for label in self.class_labels:
			dataset_label_i_file_paths = self.get_dataset_file_paths_by_label(label, dataset_name)
			dataset_file_paths_dict[label] = dataset_label_i_file_paths
		return dataset_file_paths_dict
	
	def get_dataset_img_arrays_and_labels_dict(self, dataset_name):
		img_arrays_dict = {}
		dataset_img_file_paths_dict = self.get_dataset_label_file_paths_dict(dataset_name)
		for class_label in self.class_labels:
			label_i_img_file_paths = dataset_img_file_paths_dict[class_label]
			label_i_img_arrays = []
			for img_file_path in label_i_img_file_paths:
				img_array = cv2.imread(img_file_path, 0)
				label_i_img_arrays.append(img_array)
			img_arrays_dict[class_label] = label_i_img_arrays
		return img_arrays_dict
	
	def one_got_encode_class_labels(self, y):
		y = y.reshape((y.shape[0], 1))
		onehot_encoder = OneHotEncoder(sparse=False)
		y = onehot_encoder.fit_transform(y)
		return y
	
	def get_dataset_images_and_labels(self, dataset, mode="gray_scale"):
		"""
		:param dataset: e.g. "training"
		:param mode: Is "gray_scale" (for frame_channels=0) by def. mode="rgb" loads images
		in rgb format (frame_channels=3).
		:return: np.ndarray with shape of (# of images, frame_height, frame_width, frame_channels)
		"""
		X_list = []
		y_list = []
		for class_label in self.class_labels:
			label_i_dir_path = os.path.join(self.data_subset_paths_dict[dataset], str(class_label))
			label_i_dir_file_names = os.listdir(label_i_dir_path)
			for img_i_name in label_i_dir_file_names:
				img_i_path = os.path.join(label_i_dir_path, img_i_name)
				if mode == "rgb":
					img_array = cv2.imread(img_i_path, cv2.COLOR_BGR2RGB)
				else:
					img_array_brg = cv2.imread(img_i_path)
					img_array = cv2.cvtColor(img_array_brg, cv2.COLOR_BGR2GRAY)
					print(f"img_array.shape: {img_array.shape}")
					img_array = np.expand_dims(img_array, axis=-1)
					print(f"img_array.shape: {img_array.shape}")
				X_list.append(img_array)
				y_list.append(class_label)
		X_array = np.asarray(X_list)
		# if max_img_width is not None:
		# 	X_array = self.trim_images_width(X_array, max_img_width)
		y_array = np.asarray(y_list)
		y_1_hot_encoded_array = self.one_got_encode_class_labels(y_array)
		return X_array, y_1_hot_encoded_array

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
	
	def gen_labelled_frames(self, X, y, frame_width):
		untrimmed_images = X
		og_img_width = X[0].shape[1]
		max_img_width = data_preprocessing.get_compatible_img_and_frame_widths(og_img_width, frame_width)
		trimmed_images = self.trim_images_width(untrimmed_images, max_img_width)
		all_X_frames = self.gen_img_frames(trimmed_images, frame_width)
		return all_X_frames, y

	def get_img_data_frames_and_labels(self, frame_width, dataset_name):
		dataset = self.get_dataset_images_and_labels(dataset_name, mode="rgb")
		X = dataset[0]
		y = dataset[1]
		all_X_frames = self.gen_img_frames(X, frame_width)
		return all_X_frames, y




if __name__ == "__main__":
	from src.utils import image_data_viz
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
	X, y = src.get_dataset_images_and_labels(dataset_dir_name, channels_rgb)
	frames, labels = src.gen_labelled_frames(X, y, frame_width)
	print(f"frames.shape: {frames.shape}")
	print(f"labels.shape: {labels.shape}")
	print(f"frames[0].shape: {frames[0].shape}")
	print(f"labels[0].shape: {labels[0].shape}")
	print(f"len(frames[0]): {len(frames[0])}")
	print(f"len(labels[0]): {len(labels[0])}")

	