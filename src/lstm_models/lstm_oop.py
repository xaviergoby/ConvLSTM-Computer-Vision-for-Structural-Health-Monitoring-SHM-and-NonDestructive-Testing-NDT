# import pandas as pd
# import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM, Dropout
# from keras.utils import to_categorical
# from sklearn.model_selection import train_test_split


def verify_and_fix_data(X, y, time_steps = None,
                        batch_size = None, features = None,
                        num_of_class_labels = None):
	"""
	:param X: Observation/measurement data
	:param y: target class labels
	:param time_steps: total number of time steps/rows of X_train
	:param batch_size: number of time steps/rows of X_train per batch/sequence/time series
	:param features: tot num of features/num of cols of X_train
	:param num_of_class_labels: tot num of class labels for classification
	:return: X_train w/ shape (time_steps, batch_size, features &
	y_train w/ shape (time_steps, num_of_class_labels)
	"""