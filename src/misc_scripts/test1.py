from keras.layers import Conv1D
from keras.models import Sequential
from keras.layers import Activation



model = Sequential()

# The length (number of times steps) of the time series sequences
num_of_time_steps = 30
# The number of features per time step e.g. x, y_train and z coordinates
num_of_features_per_time_step = 3

model.add(Conv1D(1, kernel_size=5, input_shape = (num_of_time_steps, num_of_features_per_time_step), strides=1, activation="relu"))
model.add(Activation('relu'))


##################################################################################


from keras.datasets import imdb
from keras.preprocessing import sequence
max_features = 10000
maxlen = 500
batch_size = 32
print('Loading data...')
(input_train, y_train), (input_test, y_test) = imdb.load_data(
num_words=max_features)
print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')
print('Pad sequences (samples x time)')
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)

##################################################################################

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ConvLSTM2D
from keras.layers import Activation, Dropout, Flatten, Dense, LeakyReLU
from keras.layers import LSTM, TimeDistributed, Lambda, BatchNormalization
from keras import optimizers
from keras import backend as K
import tensorflow as tf
from matplotlib import pyplot as plt
from IPython.display import clear_output
import keras
# from keras import models
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model, Sequential
from keras.layers import TimeDistributed
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model, Sequential


frames_per_seq = 10
img_width = 4101
img_height = 247
train_data_dir = '/training'
validation_data_dir = 'validation'


multiplier = 1
num_classes = 9
nb_train_samples = multiplier*num_classes*70
nb_validation_samples = multiplier*num_classes*20
epochs = 50
batch_size = 10


if K.image_data_format() == 'channels_first':
	# cnn_model_input_tensor_shape = (batch_size, n_channels, image_height, image_width)
	# cnn_model_input_tensor_shape = (frames_per_seq, 3, img_width, img_height)
	# cnn_model_input_tensor_shape = (None, 3, img_width, img_height)
	# cnn_model_input_tensor_shape = (3, img_width, img_height)
	input_shape = (1, img_width, img_height, None)
else:
	# (batch_size, image_height, image_width, n_channels)
	# cnn_model_input_tensor_shape = (frames_per_seq, img_width, img_height, 3)
	# cnn_model_input_tensor_shape = (None, img_width, img_height, 3)
	# cnn_model_input_tensor_shape = (img_width, img_height, 3)
	input_shape = (None, img_width, img_height, 1)

input_tensor = Input(shape=input_shape)

conv1 = TimeDistributed(Conv2D(8, kernel_size=(3, 3), activation="relu"))(input_tensor)
mp1 = TimeDistributed(MaxPooling2D((3, 3), padding='same'))(conv1)
do1 = TimeDistributed(Dropout(0.5))(mp1)

conv2 = TimeDistributed(Conv2D(16, kernel_size=(3, 3), activation="relu"))(do1)
mp2 = TimeDistributed(MaxPooling2D((3, 3), padding='same'))(conv2)
do2 = TimeDistributed(Dropout(0.5))(mp2)

conv3 = TimeDistributed(Conv2D(32, kernel_size=(3, 3), activation="relu"))(do2)
mp3 = TimeDistributed(MaxPooling2D((3, 3), padding='same'))(conv3)
do3 = TimeDistributed(Dropout(0.5))(mp3)

flat = TimeDistributed(Flatten())(do3)
# LeakyReLU
# MaxPooling2D
# Dropout
# conv2 = TimeDistributed(Conv2D(8, kernel_size=(3, 3), activation='relu'))(conv1)
# pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2)
# flat = TimeDistributed(Flatten())(pool1)

lstm = LSTM(50, return_sequences=True, activation='tanh')(flat)
d1 = Dense(128, activation="relu")(lstm)
do4 = Dropout(0.5)(d1)
d2 = Dense(16, activation="relu")(do4)
do5 = Dropout(0.5)(d2)
d3 = Dense(num_classes, activation="softmax")(do5)

model = Model(inputs=[input_tensor], outputs=d3)

model.compile(loss='categorical_crossentropy', metrics=['categorical_accuracy'], optimizer="adam")

print(model.summary)
from keras.utils import plot_model
plot_model(model, to_file='conv_lstm_model.png')








# image_data_format	Tensor shape
# channels_last	(batch_size, image_height, image_width, n_channels)
# channels_first	(batch_size, n_channels, image_height, image_width)



# from keras.layers import Input, Conv2D
# input_tensor = Input((64, 64, 3))  # 64x64 pixels, 3 frame_channels
# conv_layer = Conv2D(filters=17, kernel_size=(3, 3))
# output_tensor = conv_layer(input_tensor)
#
# In [1]: conv_layer.cnn_model_input_tensor_shape
# Out[1]: (None, 64, 64, 3)
#
# In [2]: conv_layer.output_shape
# Out[2]: (None, 62, 62, 17)