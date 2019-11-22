import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import cv2
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from src.load_image_data import ImageDataSource
import tensorflow as tf
from sklearn.metrics import confusion_matrix


class ImagePredictor:

	def __init__(self, model_name):
		self.model_name = model_name
		self.saved_trained_model_dir_path = os.path.join(os.path.dirname(os.getcwd()), r"saved_trained_model_weights")
		self.saved_trained_model_full_path = os.path.join(self.saved_trained_model_dir_path, self.model_name)
		self.test_images_dir_path = os.path.join(os.path.dirname(os.getcwd()), r"data\images\test")
		self.model = load_model(self.saved_trained_model_full_path)
		self.model_input_shape = self.model.input.shape.as_list()
		self.image_width = self.model_input_shape[1]
		self.image_height = self.model_input_shape[2]
		self.image_channels = self.model_input_shape[3]


	def _load_image(self, full_img_path):
		# """
		# :param full_img_path: the full path to the image file
		# e.g. C:\Users\JohnDoe\LSTMforSHM\data\images\test\1\Cr_percent_10_crack_angle_15_2.png
		# :return: A PIL Image instance.
		# """
		img = image.load_img(full_img_path, target_size=(self.image_width, self.image_height))
		return img

	def _img_transformer(self, img):
		img_array = image.img_to_array(img)
		img_array = img_array / 255
		img_array = np.expand_dims(img_array, axis = 0)
		ready_img_array = np.vstack([img_array])
		return ready_img_array

	def _get_pred_prob_pcts(self, img_array):
		"""
		:param img_array: array representation of an image
		:return: an array of the prob % of each class
		"""
		class_probs = self.model.predict(img_array)
		class_pred_prob_pcts = class_probs * 100
		return class_pred_prob_pcts

	def get_pred_prob_pcts(self, img_array):
		return self._get_pred_prob_pcts(img_array)

	def _get_pred_class_label(self, img_array):
		class_pred_label = self.model.predict_classes(img_array)
		class_pred_label = int(class_pred_label)
		return class_pred_label

	def get_pred_class_label(self, img_array):
		return self._get_pred_class_label(img_array)

	def _print_info(self, img_file_name, class_pred_prob_pcts, class_pred_label):
		print("\nImage file name: {0}\nPredicted class label probabilities: {1}"
		      "\nPredicted class label: {2}".format(img_file_name, class_pred_prob_pcts, class_pred_label))

	def get_pred_label_by_img_file_path(self, file_path, print_info = False):

		img = self._load_image(file_path)
		img_array = self._img_transformer(img)
		class_pred_prob_pcts = self._get_pred_prob_pcts(img_array)
		class_pred_label = self._get_pred_class_label(img_array)
		if print_info is True:
			self._print_info(file_path, class_pred_prob_pcts, class_pred_label)
		return class_pred_label, class_pred_prob_pcts, file_path

	def _predict_by_label(self, label):
		"""
		:param label: int or str of class label for which
		predictions of all associated images are to be made
		:return: a list containing the all the predicted class labels
		"""
		class_preds_list = []
		file_paths = ImageDataSource().get_test_image_file_paths_by_label(label)
		for file_path in file_paths:
			image_class_pred_and_path = self.get_pred_label_by_img_file_path(file_path)
			class_preds_list.append(image_class_pred_and_path)
		return class_preds_list

	def predict_by_label(self, label):
		"""
		:param label: int or str of class label for which
		predictions of all associated images are to be made
		:return: a numpy array containing the all the predicted class labels
		"""
		return np.asarray(self._predict_by_label(label))

	def get_all_test_class_labels(self):
		return  ImageDataSource().get_all_test_class_labels()

	def _get_all_test_image_preds(self):
		"""
		This function handles the proces of using a trained model to generate predictions of the class
		labels of each and everyone of the images contained within the test images directory
		:return: 2-tuple type consisting of a list of predicted class labels and true class labels
		"""
		class_labels_list = self.get_all_test_class_labels()
		true_class_labels_list = []
		pred_class_labels_list = []
		for class_label in class_labels_list:
			class_label_image_file_paths = ImageDataSource().get_test_image_file_paths_by_label(class_label)
			for image_file_path in class_label_image_file_paths:
				true_class_label = class_label
				true_class_labels_list.append(true_class_label)
				pred_class_label = self.get_pred_label_by_img_file_path(image_file_path)[0]
				pred_class_labels_list.append(pred_class_label)
		return pred_class_labels_list, true_class_labels_list

	def get_all_test_image_preds(self):
		return self._get_all_test_image_preds()

	def get_confusion_matrix(self, true_class_labels = None, pred_class_labels = None):
		"""
		:param true_class_labels: list of true class labels
		:param pred_class_labels: list of predicted class labels
		:return: confusion mnatrix
		"""
		if true_class_labels is None and pred_class_labels is None:
			true_and_pred_class_labels = self.get_all_test_image_preds()
			y_true = np.asarray(true_and_pred_class_labels[1])
			y_pred = np.asarray(true_and_pred_class_labels[0])
		else:
			y_true = np.asarray(true_class_labels)
			y_pred = np.asarray(pred_class_labels)
		conf_mat = confusion_matrix(y_true, y_pred)
		return conf_mat


if __name__ == "__main__":
	pred_obj = ImagePredictor("predictor.h5")
	conf_mat = pred_obj.get_confusion_matrix()
	from src import viz_tool
	viz_tool.VisTool().plot_confusion_matrix(conf_mat, pred_obj.get_all_test_class_labels())