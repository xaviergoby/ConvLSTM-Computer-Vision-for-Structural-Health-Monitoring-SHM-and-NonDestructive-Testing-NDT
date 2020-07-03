from keras.models import Sequential
from keras.layers import LeakyReLU, Dropout, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D

x_filter = 3
y_filter = 3
stride = 1
x_pool = 2
y_pool = 2
do_rate = 0.25

def build_simple_cnn_feature_extractor_seq_model(input_shape):
    """
    :param input_shape: e.g. (frame_height, frame_width, frame_channels)
    :return:
    """
    model = Sequential()
    model.add(Conv2D(16, (x_filter, y_filter), strides = (stride, stride), activation='relu', padding='same',kernel_initializer='random_uniform', input_shape=input_shape))
    model.add(MaxPooling2D((x_pool, y_pool), padding='same'))
    model.add(Dropout(do_rate))
    model.add(Conv2D(32, (x_filter, y_filter), strides = (stride, stride), activation='relu', padding='same'))
    model.add(MaxPooling2D((x_pool, y_pool), padding='same'))
    model.add(Dropout(do_rate))
    model.add(Conv2D(64, (x_filter, y_filter), strides = (stride, stride), activation='relu', padding='same'))
    model.add(MaxPooling2D((x_pool, y_pool), padding='same'))
    model.add(Dropout(do_rate))
    model.add(Conv2D(128, (x_filter, y_filter), strides = (stride, stride), activation='relu', padding='same'))
    model.add(MaxPooling2D((x_pool, y_pool), padding='same'))
    model.add(Dropout(do_rate))   
    model.add(Conv2D(64, (x_filter, y_filter), strides = (stride, stride), activation='relu', padding='same'))
    model.add(MaxPooling2D((x_pool, y_pool), padding='same'))
    model.add(Dropout(do_rate))
    model.add(Conv2D(32, (x_filter, y_filter), strides = (stride, stride), activation='relu', padding='same'))
    model.add(MaxPooling2D((x_pool, y_pool), padding='same'))
    model.add(Dropout(do_rate))
    model.add(Conv2D(16, (x_filter, y_filter), strides = (stride, stride), activation='relu', padding='same'))
    model.add(MaxPooling2D((x_pool, y_pool), padding='same'))
    model.add(Dropout(do_rate))
    #model.add(Flatten())
    model.add(GlobalAveragePooling2D())
    return model