from keras.layers import TimeDistributed
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model, Sequential
from load_image_data import ImageDataSource
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import settings
from src.cnn_models import cnn_models_collection
from src.lstm_models import lstm_models_collection

# Frame shape dimensions for input tensor shapes
height = settings.FRAME_HEIGHT
width = settings.FRAME_WIDTH
channels = settings.FRAME_CHANNELS
# Additional (shape) dimensions for input tensor shapes
num_classes = settings.NUM_CLASSES
batches_num = settings.NUM_IMGS  # number of batches
frames_num = settings.NUM_FRAMES  # number of sequential samples

# CNN 2D Seq Model
input_shape = settings.CNN_2D_INPUT_TENSOR_SHAPE
cnn = cnn_models_collection.build_simple_cnn_feature_extractor_model(input_shape)

# TimeDistributed Func API Model
video_input_tensor_shape = settings.TIME_DISTRIBUTED_MODEL_INPUT_TENSOR_SHAPE
video_input_tensor = Input(shape=video_input_tensor_shape)
td_model_output = TimeDistributed(cnn)(video_input_tensor)

# LSTM Model
fc_lstm_model_output = lstm_models_collection.get_simple_single_layer_fc_lstm_func_api_model_output(td_model_output, num_classes)

# In the functional API, given some input tensor(s) and output tensor(s), I can instantiate a Model
# which model will include all layers required in the computation of (output) b given (input) a.
model = Model(inputs=video_input_tensor, outputs=fc_lstm_model_output)

# Compilation
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer="adam")

print(model.summary())

# Retrieve img frames and img class labels dataset
src = ImageDataSource()
frames_and_labels = src.get_img_data_frames_and_labels()
frames = frames_and_labels[0]
labels = frames_and_labels[1]
X = frames
y = labels

# Partition dataset
test_split = settings.TEST_DATASET_FRACTION
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_split, shuffle=True)

# 16
#
# In keras, fit() is much similar to sklearn's fit method, where you pass array of features as x values and target as y values.
# You pass your whole dataset at once in fit method. Also, use it if you can load whole data into your memory (small dataset).
history = model.fit(X_train, y_train, epochs=1, verbose=1, batch_size=10, validation_split=0.05)


# train_steps_per_epoch = X_train.shape[0] // batch_size
# val_steps = X_val.shape[0] // batch_size
# data_gen = ImageDataGenerator(rescale=1 / 255.)
# training_generator = data_gen.flow(X_train, y_train, batch_size=batch_size, shuffle=False)
# validation_generator = data_gen.flow(X_val, y_val, batch_size=batch_size, shuffle=False)
# history = model.fit_generator(training_generator,
#                               shuffle=False,
#                               epochs=epochs,
#                               validation_data=validation_generator,
#                               validation_steps=val_steps)



plt.plot(history.history['accuracy'], 'bo')
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'], 'bo')
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# eval_res = model.evaluate(X_val, y_val)
# print(f"eval_res:\n{eval_res}")
preds = model.predict(X_val)
print(f"preds:\n{preds}")
