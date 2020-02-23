# from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
# from keras.models import Sequential
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from load_image_data import ImageDataSource

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")
# x_train = np.expand_dims(x_train, axis=-1)
train_split = 0.8
val_split = 1 - train_split