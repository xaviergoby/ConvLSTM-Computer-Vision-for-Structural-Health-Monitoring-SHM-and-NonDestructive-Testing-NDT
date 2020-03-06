from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense, LeakyReLU




def build_simple_cnn_feature_extractor_model(input_shape):
	"""
	:param input_shape: e.g. (height, width, channels)
	:return:
	"""
	model = Sequential()
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	return model


def build_vincents_fc_cnn_trainer_model(input_shape, num_classes):
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