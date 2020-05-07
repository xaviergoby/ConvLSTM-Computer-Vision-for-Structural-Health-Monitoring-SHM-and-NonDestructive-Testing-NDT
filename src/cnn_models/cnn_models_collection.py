from keras.models import Sequential
from keras.layers import LeakyReLU
from keras.layers import Dense, Dropout, Input
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.models import Model

def build_simple_cnn_feature_extractor_seq_model(input_shape):
    """
    :param input_shape: e.g. (frame_height, frame_width, frame_channels)
    :return:
    """
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same',kernel_initializer='random_uniform', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    return model

# Creating a simple CNN model in keras using functional API
def build_simple_fc_cnn_func_api_model(input_shape, num_classes):
	img_inputs = Input(shape=input_shape)
	conv_1 = Conv2D(32, (3, 3), activation='relu')(img_inputs)
	maxpool_1 = MaxPooling2D((2, 2))(conv_1)
	conv_2 = Conv2D(64, (3, 3), activation='relu')(maxpool_1)
	maxpool_2 = MaxPooling2D((2, 2))(conv_2)
	conv_3 = Conv2D(64, (3, 3), activation='relu')(maxpool_2)
	flatten = Flatten()(conv_3)
	dense_1 = Dense(64, activation='relu')(flatten)
	output = Dense(num_classes, activation='softmax')(dense_1)
	model = Model(inputs=img_inputs, outputs=output)
	return model

def build_vincents_fc_cnn_seq_trainer_model(input_shape, num_classes):
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