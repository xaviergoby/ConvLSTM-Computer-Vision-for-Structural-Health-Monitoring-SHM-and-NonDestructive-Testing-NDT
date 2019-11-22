import pandas as pd
import matplotlib.pyplot as plt
import os

# img_path = "data/SPIE2019/All/test/Sample2.png

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

		self.test_labels_list = [int(label) for label in os.listdir(self.test_dir_full_path)]

	def get_all_test_class_labels(self):
		return self.test_labels_list


	def crawl_test_dir(self):
		for dirname, dirnames, filenames in os.walk(self.test_dir_full_path):
			print(self.test_dir_full_path)
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


if __name__ == "__main__":
	# img_file_names = LSTMDataSource().get_image_file_names_by_label(0)
	# x = ImageDataSource().get_test_image_file_paths_by_label(0)
	x = os.path.join(os.path.dirname(os.getcwd()), r"data\images\test")