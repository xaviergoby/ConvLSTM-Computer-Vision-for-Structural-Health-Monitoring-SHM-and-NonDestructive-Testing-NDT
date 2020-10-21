import os
import configs_and_settings
import cv2
import numpy as np
from src.utils import file_manipulation_tools


class ImageDataSetLoader:
	
	def __init__(self, dataset_dir_name):
		"""
		:param dataset_dir_name: str of the name of the dir/folder containing the dataset
		"""
		self.dataset_dir_name = dataset_dir_name
		self.dataset_dir_path = os.path.join(configs_and_settings.IMAGES_DIR, self.dataset_dir_name)
		self.training_dir_path = os.path.join(self.dataset_dir_path, "training")
		self.validation_dir_path = os.path.join(self.dataset_dir_path, "validation")
		self.test_dir_path = os.path.join(self.dataset_dir_path, "test")
		self.data_subset_paths_dict = {"training": self.training_dir_path,
		                               "validation": self.validation_dir_path,
		                               "test": self.test_dir_path}
		
		self.class_labels = list(map(int, file_manipulation_tools.get_file_folder_names_in_dir(self.training_dir_path)))
		self.num_class_labels = len(self.class_labels)
		self.num_training_images_per_label = file_manipulation_tools.get_num_files_in_dir(
			os.path.join(self.training_dir_path, str(self.class_labels[0])))
		self.num_validation_images_per_label = file_manipulation_tools.get_num_files_in_dir(
			os.path.join(self.validation_dir_path, str(self.class_labels[0])))
		self.num_test_images_per_label = file_manipulation_tools.get_num_files_in_dir(
			os.path.join(self.test_dir_path, str(self.class_labels[0])))
		self.tot_num_training_images = self.num_class_labels * self.num_training_images_per_label
		self.tot_num_validation_images = self.num_class_labels * self.num_validation_images_per_label
		self.tot_num_test_images = self.num_class_labels * self.num_test_images_per_label
		self.dataset_images = None
		self.dataset_labels = None
		self.dataset_partitions_dict = None
		
		self._load_dataset()
	
	def __getitem__(self, index):
		return self.dataset_partitions_dict[index]
	
	def _load_dataset(self):
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
		images_list = []
		labels_list = []
		dataset_partitions_dict = {}
		for dataset_split_i_name in list(self.data_subset_paths_dict.keys()):
			dataset_split_i = self._load_dataset_split(dataset_split_name=dataset_split_i_name)
			dataset_split_i_inputs = dataset_split_i[0]
			dataset_split_i_targets = dataset_split_i[1]
			images_list.append(dataset_split_i_inputs)
			labels_list.append(dataset_split_i_targets)
			dataset_partitions_dict[dataset_split_i_name] = (dataset_split_i_inputs, dataset_split_i_targets)
		self.dataset_images = images_list
		self.dataset_labels = labels_list
		self.dataset_partitions_dict = dataset_partitions_dict
	
	def _load_dataset_split(self, dataset_split_name):
		"""
		:param dataset_split_name: e.g. "training"
		:param mode: Is "rgb" (for frame_channels=0) by def. mode="gray_scale" loads dataset_split_name
		in gray scale format (frame_channels=1).
		:return: np.ndarray with shape of (# of image_datasets, frame_height, frame_width, frame_channels)
		"""
		images_list = []
		labels_list = []
		for class_label_i in self.class_labels:
			label_i_dir_path = os.path.join(self.data_subset_paths_dict[dataset_split_name], str(class_label_i))
			label_i_dir_file_names = os.listdir(label_i_dir_path)
			for img_i_name in label_i_dir_file_names:
				img_i_path = os.path.join(label_i_dir_path, img_i_name)
				img_array = cv2.imread(img_i_path, cv2.COLOR_BGR2RGB)
				images_list.append(img_array)
				labels_list.append(class_label_i)
		images_array = np.asarray(images_list)
		labels_array = np.asarray(labels_list)
		return images_array, labels_array
	
	@property
	def training_dataset(self):
		return self.dataset_partitions_dict["training"]
	
	@property
	def validation_dataset(self):
		return self.dataset_partitions_dict["validation"]
	
	@property
	def test_dataset(self):
		return self.dataset_partitions_dict["test"]

	


if __name__ == "__main__":
	data_dir_name = "scenario_1"
	# dataset_dir_name = "training"
	dataset = ImageDataSetLoader(data_dir_name)
	dataset_images = dataset.dataset_images
	dataset_class_labels = dataset.dataset_labels
	dataset_partitions_dict = dataset.dataset_partitions_dict
	print(f"type(dataset_images): {type(dataset_images)}")
	print(f"type(dataset_class_labels): {type(dataset_class_labels)}")
	print(f"type(dataset_partitions_dict): {type(dataset_partitions_dict)}")
	print(f"len(dataset_images): {len(dataset_images)}")
	print(f"len(dataset_class_labels): {len(dataset_class_labels)}")
	print(dataset["training"])
	val_data = dataset.validation_dataset
	train_data = dataset.training_dataset
	test_data = dataset.test_dataset
	print(f"type(train_data): {type(train_data)}")
	print(f"type(train_data[0]): {type(train_data[0])}")
	print(f"type(val_data): {type(val_data)}")
	print(f"type(test_data): {type(test_data)}")
	
	# input_dataset_splits, target_dataset_splits = dataset.load_dataset(mode="rgb")
	# print(f"src.class_labels: {dataset.class_labels}")
	# print(f"src.num_class_labels: {dataset.num_class_labels}")
	# print(f"src.num_training_images: {dataset.num_training_images_per_label}")
	# print(f"src.num_validation_images: {dataset.num_validation_images_per_label}")
	# print(f"src.num_test_images: {dataset.num_test_images_per_label}")
	# img_width = 4101
	# frame_width = 25
	# max_img_width = data_preprocessing.get_compatible_img_and_frame_widths(img_width, frame_width)
	# channels_rgb = "rgb"
	# channels_gray = "gray_scale"
	# # X, y = src._load_dataset_split(dataset_dir_name, channels_rgb)
	# input_dataset_splits, target_dataset_splits = dataset.load_dataset(mode="rgb")
	# X = input_dataset_splits[0]
	# y = target_dataset_splits[0]
	# frames, labels = dataset.gen_labelled_frames_batches(X, y, frame_width)
	# print(f"frames.shape: {frames.shape}")
	# print(f"labels.shape: {labels.shape}")
	# print(f"frames[0].shape: {frames[0].shape}")
	# print(f"labels[0].shape: {labels[0].shape}")
	# print(f"len(frames[0]): {len(frames[0])}")
	# print(f"len(labels[0]): {len(labels[0])}")
