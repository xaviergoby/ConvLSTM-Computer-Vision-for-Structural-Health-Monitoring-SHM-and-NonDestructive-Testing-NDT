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
img_width = 400
img_height = 247


# func to partition large images into multiple smaller) frames which collectively constitute 1 sequence
# func to


if K.image_data_format() == 'channels_first':
	# (channels, rows, cols) so: rows = conv_dim1 & cols = conv_dim2
	# (channels, conv_dim1, conv_dim2, conv_dim3)
	input_shape = (1, img_width, img_height, frames_per_seq)
else:
	# (rows, cols, channels)
	# (conv_dim1, conv_dim2, conv_dim3, channels)
	input_shape = (frames_per_seq, img_width, img_height, 1)

seq_of_frames = Input(shape=input_shape)

conv1 = TimeDistributed(Conv2D(8, kernel_size=(3, 3), activation='relu'))(seq_of_frames)
conv2 = TimeDistributed(Conv2D(8, kernel_size=(3, 3), activation='relu'))(conv1)
pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2)
flat = TimeDistributed(Flatten())(pool1)

lstm = LSTM(50, return_sequences=True, activation='tanh')(flat)
op = TimeDistributed(Dense(9))(lstm)
model = Model(inputs=[seq_of_frames], outputs=op)

model.compile(loss="binary_crossentropy", metrics=["acc"], optimizer="adam")
print(model.summary)
from keras.utils import plot_model
plot_model(model, to_file='model.png')


# X = array(X).reshape(n_patterns, size, size, size, 1)

# model.fit(X, y, batch_size=32, epochs=1)