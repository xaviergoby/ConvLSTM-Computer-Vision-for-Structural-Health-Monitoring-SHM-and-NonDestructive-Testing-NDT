from keras.layers import Dropout
import numpy as np
from keras.layers import TimeDistributed
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import settings
from sklearn.preprocessing import OneHotEncoder
from load_image_data import ImageDataSource
batch_size = 1
# img_width = 4101
img_height = 4101
# img_height = 247
img_width = 247
channels = 1
num_classes = 9


# if K.image_data_format() == 'channels_first':
	# input_shape = (batch_size, n_channels, image_height, image_width)
	# input_shape = (frames_per_seq, 3, img_width, img_height)
	# input_shape = (None, 3, img_width, img_height)
	# input_shape = (3, img_width, img_height)
	# input_shape = (channels, img_width, img_height, None)
	# input_shape = (channels, img_width, img_height, 2)
# 	input_shape = (channels, img_width, img_height)
# else:
	# (batch_size, image_height, image_width, n_channels)
	# input_shape = (frames_per_seq, img_width, img_height, 3)
	# input_shape = (None, img_width, img_height, 3)
	# input_shape = (img_width, img_height, 3)
	# input_shape = (None, img_width, img_height, channels)
	# input_shape = (2, img_width, img_height, channels)
	# input_shape = (img_width, img_height, channels)

batches_num = 90    # number of batches
frames_num = 164    # number of sequential samples
height = 247
width = 25
channels = 1
input_tensor_shape = (frames_num, height, width, channels)
# input_shape = (channels, img_width, img_height)
# input_shape = (img_width, img_height, channels)
# input_shape = (number_of_frames, img_width, img_height, channels)
# input_shape = (img_width, img_height, channels)
input_tensor = Input(shape=input_tensor_shape)

conv1 = TimeDistributed(Conv2D(16, (2, 2), activation="relu"))(input_tensor)
# mp1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv1)
mp1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same'))(conv1)
do1 = TimeDistributed(Dropout(0.5))(mp1)

conv2 = TimeDistributed(Conv2D(32, (2, 2), activation="relu"))(do1)
# mp2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2)
mp2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same'))(conv2)
do2 = TimeDistributed(Dropout(0.5))(mp2)

conv3 = TimeDistributed(Conv2D(64, (2, 2), activation="relu"))(do2)
# mp3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv3)
mp3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same'))(conv3)
# do3 = TimeDistributed(Dropout(0.5))(mp3)

flat = TimeDistributed(Flatten())(mp3)
# flat = TimeDistributed(Flatten())(do3)

lstm = LSTM(256, return_sequences=True, activation='tanh')(flat)
d1 = Dense(128, activation="relu")(lstm)
do4 = Dropout(0.5)(d1)
d2 = Dense(16, activation="relu")(do4)
do5 = Dropout(0.5)(d2)
d3 = Dense(num_classes, activation="softmax")(do5)

model = Model(inputs=input_tensor, outputs=d3)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer="adam")



img_data_src = ImageDataSource()
img_dataset = img_data_src.get_dataset()
X = img_dataset[0]
# X = X.reshape((X.shape[0], 1, X.shape[1], X.shape[2], X.shape[-1]))
y = img_dataset[1]
y = y.reshape((y.shape[0], 1))
# label_encoder = LabelEncoder()
onehot_encoder = OneHotEncoder(sparse=False)
y = onehot_encoder.fit_transform(y)

num_tot_samples = X.shape[0]
train_split = 0.8
# val_split = 0.2
epochs = 10
batch_size = 1

from sklearn.model_selection import train_test_split
val_split = 0.2
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_split, stratify=y)
# X = np.concatenate((X_train, X_val))
# y = np.concatenate((y_train, y_val))

train_steps_per_epoch = X_train.shape[0] // batch_size
val_steps = X_val.shape[0] // batch_size

data_gen = ImageDataGenerator(rescale=1 / 255.)
training_generator = data_gen.flow(X_train, y_train, batch_size=batch_size)
validation_generator = data_gen.flow(X_val, y_val, batch_size=batch_size)

history = model.fit_generator(training_generator,
							  steps_per_epoch=train_steps_per_epoch,
							  epochs=epochs,
							  validation_data=validation_generator,
							  validation_steps=val_steps)


print(model.summary)
# from keras.utils import plot_model
# plot_model(model, to_file='model.png')

