from keras.layers import TimeDistributed
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Activation, Flatten, Dense
from keras.utils import plot_model

# source: https://gist.github.com/HTLife/ca0a7d48bd9a3192cf8d3c8b1347e8dd

def defModel():
	model = Sequential()
	# Izda.add(TimeDistributed(
	#    Convolution2D(40,3,3,border_mode='same'), input_shape=(sequence_lengths, 1,8,10)))
	model.add(
		TimeDistributed(
			Conv2D(32, (7, 7), padding='same', strides=2),
			input_shape=(None, 540, 960, 2)))
	model.add(Activation('relu'))

	model.add(TimeDistributed(Conv2D(64, (5, 5), padding='same', strides=2)))
	model.add(Activation('relu'))

	# model.add(TimeDistributed(MaxPooling2D((2,2), data_format = 'channels_first', name='pool1')))

	model.add(TimeDistributed(Conv2D(128, (5, 5), padding='same', strides=2)))
	model.add(Activation('relu'))

	model.add(TimeDistributed(Conv2D(128, (3, 3), padding='same')))
	model.add(Activation('relu'))

	model.add(TimeDistributed(Conv2D(256, (3, 3), padding='same', strides=2)))
	model.add(Activation('relu'))

	model.add(TimeDistributed(Conv2D(256, (3, 3), padding='same')))
	model.add(Activation('relu'))

	model.add(TimeDistributed(Conv2D(256, (3, 3), padding='same', strides=2)))
	model.add(Activation('relu'))

	model.add(TimeDistributed(Conv2D(256, (3, 3), padding='same')))
	model.add(Activation('relu'))

	model.add(TimeDistributed(Conv2D(512, (3, 3), padding='same', strides=2)))
	model.add(Activation('relu'))
	# model.add(TimeDistributed(MaxPooling2D((2,2), data_format = 'channels_first', name='pool1')))

	# model.add(TimeDistributed(Conv2D(32, (1, 1), data_format = 'channels_first')))
	# model.add(Activation('relu'))

	model.add(TimeDistributed(Flatten()))

	# model.add(TimeDistributed(Dense(512, name="first_dense" )))

	# model.add(LSTM(num_classes, return_sequences=True))
	model.add(LSTM(512, return_sequences=True))
	model.add(LSTM(512))
	model.add(Dense(128))
	model.add(Dense(3))

	model.compile(loss='mean_squared_error', optimizer='adam')  # ,
	# metrics=['accuracy'])
	plot_model(model, to_file='model/model.png')
	plot_model(model, to_file='model/model_detail.png', show_shapes=True)
	return model