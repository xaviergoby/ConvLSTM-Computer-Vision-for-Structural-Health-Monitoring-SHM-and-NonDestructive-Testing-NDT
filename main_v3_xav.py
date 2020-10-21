import keras
from keras.layers import TimeDistributed
from keras.layers import Input
from keras.models import Model
from src.data_loading_tools.image_dataset_loaders import image_data_handler
import matplotlib.pyplot as plt
from src.cnn_models import cnn_models_collection
from src.lstm_models import lstm_models_collection
from src.utils import model_saving_funcs
from src.utils import data_preprocessing

# assign the name (str type) of the folder (which you should have placed in the folder
# located @LSTMforSHM/data/image_datasets) containing your dataset of image_datasets
# to the variable dataset_dir_name! E.g. dataset_dir_name = "scenario_1"
dataset_dir_name = "scenario_1"
train_dataset_dir_name = "training"
val_dataset_dir_name = "validation"
test_dataset_dir_name = "test"

img_data_src = image_data_handler.ImageDataHandler(dataset_dir_name)

# Frame shape dimensions for input tensor shapes
frame_width = 25
frame_height = 247
# channels = 3
channels = 1
# img_colour_format = "rgb"
img_colour_format = "gray_scale"

# Retrieve original/raw image_datasets & class labels of training, validation and test datasets
og_train_images, og_train_labels = img_data_src.get_dataset_images_and_labels(train_dataset_dir_name, img_colour_format)
og_val_images, og_val_labels = img_data_src.get_dataset_images_and_labels(val_dataset_dir_name, img_colour_format)
og_test_images, og_test_labels = img_data_src.get_dataset_images_and_labels(test_dataset_dir_name, img_colour_format)
# Get/generate modified/altered/preprocessed img frames and class labels of training, validation and test datasets
X_train, y_train = img_data_src.gen_labelled_frames_batches(og_train_images, og_train_labels, frame_width)
X_val, y_val = img_data_src.gen_labelled_frames_batches(og_val_images, og_val_labels, frame_width)
X_test, y_test = img_data_src.gen_labelled_frames_batches(og_test_images, og_test_labels, frame_width)


# Additional (shape) dimensions for input tensor shapes
num_classes = img_data_src.num_class_labels
batches_num = X_train.shape[0]
num_frames = X_train.shape[1]

# CNN 2D Seq Model
cnn_model_input_tensor_shape = (frame_height, frame_width, channels)
cnn_model = cnn_models_collection.build_simple_cnn_feature_extractor_seq_model(cnn_model_input_tensor_shape)

# TimeDistributed Func API Model
td_video_input_tensor_shape = (num_frames, frame_height, frame_width, channels)
td_video_input_tensor = Input(shape=td_video_input_tensor_shape)
td_model_output = TimeDistributed(cnn_model)(td_video_input_tensor)

# LSTM Model
fc_lstm_model_output = lstm_models_collection.get_simple_single_layer_fc_lstm_func_api_model_output(td_model_output, num_classes)

# In the functional API, given some input tensor(s) and output tensor(s), I can instantiate a Model
# which conv_lstm_model will include all layers required in the computation of (output) b given (input) a.
conv_lstm_model = Model(inputs=td_video_input_tensor, outputs=fc_lstm_model_output)

# Compilation
lr = 0.001
RMSprop_opt = keras.optimizers.RMSprop(learning_rate=lr)
adam_optimizer = keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
conv_lstm_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=RMSprop_opt)

# Print a summary of your (compiled) models constituent layers, hyper-parameters, parameters etc...
print(conv_lstm_model.summary())

# In keras, fit() is much similar to sklearn's fit method, where you pass array of features as x values and target as y_train values.
# You pass your whole dataset at once in fit method. Also, use it if you can load whole data into your memory (small dataset).
# 1660 Ti GPU Memory compatible batch sizes: 1, 2. 4
bs = 4
num_epochs = 20
# history = conv_lstm_model.fit(X_train, y_train, epochs=num_epochs, verbose=1, batch_size=bs, validation_split=0.05)
history = conv_lstm_model.fit(X_train, y_train, epochs=num_epochs, verbose=1, batch_size=bs, validation_data=(X_val, y_val))

# Visualising the performance by plotting the training history
plt.plot(history.history['accuracy'], 'g', label="training accuracy")
plt.plot(history.history['val_accuracy'], 'r', label="validation accuracy")
plt.title('conv_lstm_model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
plt.legend(loc='upper left')
plt.show()
# Visualising the history of the loss score
plt.plot(history.history['loss'], 'g', label="training loss")
plt.plot(history.history['val_loss'], 'r', label="validation loss")
plt.title('conv_lstm_model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
plt.legend(loc='upper left')
plt.show()

# Convert the trained model into dot format, save it and visualise
# the architecture of the network/model
model_saving_funcs.save_model_arch_plot(conv_lstm_model, "conv_lstm_model")

















# adam lr = 0.001
# 96s 229ms/step - loss: 1.8829 - accuracy: 0.1905 - val_loss: 1.7930 - val_accuracy: 0.1667
# 96s 229ms/step - loss: 1.8214 - accuracy: 0.1690 - val_loss: 1.7918 - val_accuracy: 0.1667
# 100s 237ms/step - loss: 1.8004 - accuracy: 0.1548 - val_loss: 1.7918 - val_accuracy: 0.1667
# 104s 247ms/step - loss: 1.7945 - accuracy: 0.1857 - val_loss: 1.7919 - val_accuracy: 0.1667
# 103s 245ms/step - loss: 1.7959 - accuracy: 0.1619 - val_loss: 1.7919 - val_accuracy: 0.1667
# 103s 245ms/step - loss: 1.7957 - accuracy: 0.1667 - val_loss: 1.7919 - val_accuracy: 0.1667
# 256ms/step - loss: 1.7963 - accuracy: 0.1571 - val_loss: 1.7919 - val_accuracy: 0.1667
# 101s 240ms/step - loss: 1.7995 - accuracy: 0.1476 - val_loss: 1.7919 - val_accuracy: 0.1667
# 102s 244ms/step - loss: 1.7953 - accuracy: 0.1595 - val_loss: 1.7918 - val_accuracy: 0.1667
# 104s 248ms/step - loss: 1.7971 - accuracy: 0.1571 - val_loss: 1.7918 - val_accuracy: 0.1667

# adam lr = 0.0001 & smaller networks (less neurons)
# 35s 83ms/step - loss: 248.7089 - accuracy: 0.1452 - val_loss: 101.6852 - val_accuracy: 0.1917
# 34s 80ms/step - loss: 174.4075 - accuracy: 0.1548 - val_loss: 77.0951 - val_accuracy: 0.1667
# 35s 83ms/step - loss: 183.0110 - accuracy: 0.1738 - val_loss: 66.6526 - val_accuracy: 0.1667
#
# adam lr = 0.001 & smaller networks (less neurons)
# 36s 85ms/step - loss: 291455477.6243 - accuracy: 0.1524 - val_loss: 255340.7504 - val_accuracy: 0.1583
# 36s 86ms/step - loss: 62963831.0192 - accuracy: 0.1310 - val_loss: 1.7932 - val_accuracy: 0.1667
#
# adam lr = 0.001 & smaller networks (less neurons) & using .fit(validation_split=0.05)
# 36s 90ms/step - loss: 424.9927 - accuracy: 0.1679 - val_loss: 0.0000e+00 - val_accuracy: 1.0000
#
#