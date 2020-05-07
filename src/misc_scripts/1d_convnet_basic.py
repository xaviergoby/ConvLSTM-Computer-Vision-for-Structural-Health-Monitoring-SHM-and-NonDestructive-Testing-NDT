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