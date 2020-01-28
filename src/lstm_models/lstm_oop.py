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
	:param time_steps: total number of time steps/rows of X
	:param batch_size: number of time steps/rows of X per batch/sequence/time series
	:param features: tot num of features/num of cols of X
	:param num_of_class_labels: tot num of class labels for classification
	:return: X w/ shape (time_steps, batch_size, features &
	y w/ shape (time_steps, num_of_class_labels)
	"""