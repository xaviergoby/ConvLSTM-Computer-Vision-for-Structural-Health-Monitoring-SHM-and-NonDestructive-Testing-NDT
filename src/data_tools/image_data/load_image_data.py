import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2

# img_path = "data/SPIE2019/All/test/Sample2.png
# The factors of 4,100. Answer : 1,2,4,5,10,20,25,41,50,82,100,164,205,410,820,1025,2050,4100,

class ImageDataSource:

	def __init__(self, train_dir_path = r"data\images\train",
				 validation_dir_path = r"data\images\validation",
				 test_dir_path = r"data\images\test"):

		self.train_dir_path = train_dir_path
		self.validation_dir_path = validation_dir_path
		self.test_dir_path = test_dir_path


		self.train_dir_full_path = os.path.join(os.path.dirname(os.getcwd()), self.train_dir_path)
		self.validation_dir_full_path = os.path.join(os.path.dirname(os.getcwd()), self.validation_dir_path)
		self.test_dir_full_path = os.path.join(os.path.dirname(os.getcwd()), self.test_dir_path)

		self.data_dir_paths_dict = {"train":self.train_dir_full_path, "validation":self.validation_dir_full_path,
		                            "test":self.test_dir_full_path}

		self.class_labels = [int(label) for label in os.listdir(self.test_dir_full_path)]

	def get_all_test_class_labels(self):
		return self.class_labels

	def crawl_data_dir(self, dir="train"):
		dir_file_path = self.data_dir_paths_dict[dir]
		for dirname, dirnames, filenames in os.walk(dir_file_path):
			# print(self.test_dir_full_path)
			# print path to all subdirectories first.
			for subdirname in dirnames:
				print(os.path.join(dirname, subdirname))

			# print path to all filenames.
			for filename in filenames:
				print(os.path.join(dirname, filename))

			# Advanced usage:
			# editing the 'dirnames' list will stop os.walk() from recursing into there.
			if '.git' in dirnames:
				# don't go into any .git directories.
				dirnames.remove('.git')

	def get_test_image_dir_path_by_label(self, label):
		label_test_images_dir_path = os.path.join(self.test_dir_full_path, str(label))
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


	def get_dataset(self, dataset_name = None):
		if dataset_name is not None:
			if dataset_name is "train":
				pass
				# self.train_dir_full_path

	def get_img_label_dir_path(self, dataset_name, abel):
		label_test_images_dir_path = os.path.join(self.test_dir_full_path, str(label))
		return label_test_images_dir_path







if __name__ == "__main__":
	# img_file_names = LSTMDataSource().get_image_file_names_by_label(0)
	# x = ImageDataSource().get_test_image_file_paths_by_label(0)
	# x = os.path.join(os.path.dirname(os.getcwd()), r"data\images\test")
	src = ImageDataSource()
	# src.get_test_imageE_file_paths_dict()
	l = src.get_test_image_file_paths_dict()
	print(len(l))
	data = src.get_test_img_arrays_and_labels_dict()
	print(data)
	print(len(data))