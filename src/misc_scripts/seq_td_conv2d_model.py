from keras.layers import TimeDistributed
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model, Sequential
from load_image_data import ImageDataSource
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import configs_and_settings

# logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard = TensorBoard(log_dir=logdir)
# tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

num_classes = 9

batches_num = 90  # number of batches
frames_num = 164  # number of sequential samples
height = 247
width = 25
channels = 1
# input_tensor_shape = (frames_num, frame_height, frame_width, frame_channels)

def build_simple_cnn_feature_extractor_model(input_shape):
	"""
	:param input_shape: e.g. (frame_height, frame_width, frame_channels)
	:return:
	"""
	cnn = Sequential()
	cnn.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape))
	cnn.add(MaxPooling2D((2, 2)))
	cnn.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	cnn.add(MaxPooling2D((2, 2)))
	cnn.add(Flatten())
	return cnn

# CNN 2D Seq Model
input_shape = configs_and_settings.CNN_2D_INPUT_TENSOR_SHAPE
cnn = build_simple_cnn_feature_extractor_model(input_shape)

# TimeDistributed Func API Model
video_input_tensor_shape = configs_and_settings.TIME_DISTRIBUTED_MODEL_INPUT_TENSOR_SHAPE
video_input_tensor = Input(shape=video_input_tensor_shape)
td_model_output = TimeDistributed(cnn)(video_input_tensor)

# LSTM Model
lstm = LSTM(128, return_sequences=False, activation='tanh')(td_model_output)
d1 = Dense(128, activation="relu")(lstm)
# do4 = Dropout(0.5)(d1)
# d2 = Dense(16, activation="relu")(do4)
d2 = Dense(16, activation="relu")(d1)
# do5 = Dropout(0.5)(d2)
# d3 = Dense(num_classes, activation="softmax")(do5)
d3 = Dense(num_classes, activation="softmax")(d2)

model = Model(inputs=video_input_tensor, outputs=d3)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer="adam")

print(model.summary())
# from keras.utils import plot_model
# plot_model(conv_lstm_model, to_file='conv_lstm_model.png')

src = ImageDataSource()
frames_and_labels = src.get_img_data_frames_and_labels()
frames = frames_and_labels[0]
labels = frames_and_labels[1]
print(f"frames.shape: {frames.shape}")
print(f"labels.shape: {labels.shape}")
print(f"frames[0].shape: {frames[0].shape}")
print(f"labels[0].shape: {labels[0].shape}")
print(f"len(frames[0]): {len(frames[0])}")
print(f"len(labels[0]): {len(labels[0])}")
X = frames
y = labels
test_split = 0.05

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_split, shuffle=True)

batch_size = 1
epochs = 10

# history = conv_lstm_model.fit(X_train, y_train, epochs=1, verbose=1, batch_size=1, validation_data=(X_val, y_val))
history = model.fit(X_train, y_train, epochs=1, verbose=1, batch_size=1, validation_split=0.05)
# history = conv_lstm_model.fit(X_train, y_train, epochs=1, verbose=1, batch_size=1, validation_data=(X_val, y_val), callbacks=[tensorboard])


# train_steps_per_epoch = X_train.shape[0] // batch_size
# val_steps = X_val.shape[0] // batch_size
# data_gen = ImageDataGenerator(rescale=1 / 255.)
# training_generator = data_gen.flow(X_train, y_train, batch_size=batch_size, shuffle=False)
# validation_generator = data_gen.flow(X_val, y_val, batch_size=batch_size, shuffle=False)
# history = conv_lstm_model.fit_generator(training_generator,
#                               shuffle=False,
#                               epochs=epochs,
#                               validation_data=validation_generator,
#                               validation_steps=val_steps)



plt.plot(history.history['accuracy'], 'bo')
plt.plot(history.history['val_accuracy'])
plt.title('conv_lstm_model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'], 'bo')
plt.plot(history.history['val_loss'])
plt.title('conv_lstm_model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# eval_res = conv_lstm_model.evaluate(X_val, y_val)
# print(f"eval_res:\n{eval_res}")
preds = model.predict(X_val)
print(f"preds:\n{preds}")
