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


def build_vincents_cnn_trainer_model(input_shape, num_classes):
	model = Sequential()
	# CNN
	model.add(Conv2D(8, (3, 3), input_shape=input_shape))
	model.add(LeakyReLU(alpha=0.01))
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	model.add(Dropout(0.5))
	model.add(Conv2D(16, (3, 3), padding='same'))
	model.add(LeakyReLU(alpha=0.01))
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	model.add(Dropout(0.5))
	model.add(Conv2D(32, (3, 3), padding='same'))
	model.add(LeakyReLU(alpha=0.01))
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	model.add(Dropout(0.5))
	model.add(Flatten())
	# MLP
	model.add(Dense(128))
	model.add(LeakyReLU(alpha=0.01))
	model.add(Dropout(0.5))
	model.add(Dense(16))
	model.add(LeakyReLU(alpha=0.01))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))

	return model

if __name__ == "__main__":

	img_width, img_height = 4101, 247
	train_data_dir = '/training'
	validation_data_dir = 'validation'

	multiplier = 1
	num_classes = 9
	nb_train_samples = multiplier * num_classes * 70
	nb_validation_samples = multiplier * num_classes * 20
	epochs = 50
	batch_size = 10

	if K.image_data_format() == 'channels_first':
		input_shape = (3, img_width, img_height)
	else:
		input_shape = (img_width, img_height, 3)

	model = build_vincents_cnn_trainer_model(input_shape, num_classes)
	adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=1e-6, amsgrad=False)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.summary()

	train_datagen = ImageDataGenerator(rescale=1. / 255)
	test_datagen = ImageDataGenerator(rescale=1. / 255)

	train_generator = train_datagen.flow_from_directory(
		train_data_dir, target_size=(img_width, img_height),
		batch_size=batch_size, color_mode='rgb', class_mode='categorical')

	validation_generator = test_datagen.flow_from_directory(
		validation_data_dir, target_size=(img_width, img_height),
		batch_size=batch_size, color_mode='rgb', class_mode='categorical')

	model.fit_generator(
		train_generator,
		steps_per_epoch=nb_train_samples // batch_size,
		epochs=epochs,
		validation_data=validation_generator,
		validation_steps=nb_validation_samples // batch_size)
