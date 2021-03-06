from keras.layers import TimeDistributed
from keras.layers import Input
from keras.models import Model
from src.data_loading_tools.image_dataset_loaders import load_image_data2
from src.data_loading_tools.image_dataset_loaders import load_image_data
from src.data_loading_tools.image_dataset_loaders import image_data_handler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import configs_and_settings
from src.cnn_models import cnn_models_collection
from src.lstm_models import lstm_models_collection
from src.utils import model_saving_funcs


data_dir_name = "scenario_1"
img_data_handler = image_data_handler.ImageDataHandler(data_dir_name)


# Frame shape dimensions for input tensor shapes
height = configs_and_settings.FRAME_HEIGHT
width = configs_and_settings.FRAME_WIDTH
channels = configs_and_settings.FRAME_CHANNELS
# Additional (shape) dimensions for input tensor shapes
num_classes = configs_and_settings.NUM_CLASSES # \\TODO change
# num_classes = img_data_src.num_class_labels \\TODO Replace above line with this one
batches_num = configs_and_settings.NUM_IMGS  # number of batches \\TODO redundant
frames_num = configs_and_settings.NUM_FRAMES  # number of sequential samples \\TODO redundant

# CNN 2D Seq Model
# 3D tensor shape which stores a single frame: (FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS)
cnn_model_input_tensor_shape = configs_and_settings.CNN_2D_INPUT_TENSOR_SHAPE
# cnn_model_input_tensor_shape =
cnn_model = cnn_models_collection.build_simple_cnn_feature_extractor_seq_model(cnn_model_input_tensor_shape)

# TimeDistributed Func API Model
# # 4D tensor shape storing a sequence of frames: (NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS)
td_video_input_tensor_shape = configs_and_settings.TIME_DISTRIBUTED_MODEL_INPUT_TENSOR_SHAPE
td_video_input_tensor = Input(shape=td_video_input_tensor_shape)
td_model_output = TimeDistributed(cnn_model)(td_video_input_tensor)

# LSTM Model
fc_lstm_model_output = lstm_models_collection.get_simple_single_layer_fc_lstm_func_api_model_output(td_model_output, num_classes)

# In the functional API, given some input tensor(s) and output tensor(s), I can instantiate a Model
# which conv_lstm_model will include all layers required in the computation of (output) b given (input) a.
conv_lstm_model = Model(inputs=td_video_input_tensor, outputs=fc_lstm_model_output)

# Compilation
conv_lstm_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer="adam")

# Print a summary of your (compiled) models constituent layers, hyper-parameters, parameters etc...
print(conv_lstm_model.summary())

# Retrieve img frames and img class labels dataset
src = load_image_data.ImageDataSource()
frames_and_labels = src.get_img_data_frames_and_labels()
frames = frames_and_labels[0]
labels = frames_and_labels[1]
X = frames
y = labels

# Partition dataset
test_split = configs_and_settings.TEST_DATASET_FRACTION
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_split, shuffle=True)

# In keras, fit() is much similar to sklearn's fit method, where you pass array of features as x values and target as y_train values.
# You pass your whole dataset at once in fit method. Also, use it if you can load whole data into your memory (small dataset).
# 1660 Ti GPU Memory compatible batch sizes: 1, 2
history = conv_lstm_model.fit(X_train, y_train, epochs=10, verbose=1, batch_size=2, validation_split=0.05)

# Visualising the performance by plotting the training history
plt.plot(history.history['accuracy'], 'bo')
plt.plot(history.history['val_accuracy'])
plt.title('conv_lstm_model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'prototyping_data'], loc='upper left')
plt.show()
# Visualising the history of the loss score
plt.plot(history.history['loss'], 'bo')
plt.plot(history.history['val_loss'])
plt.title('conv_lstm_model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'prototyping_data'], loc='upper left')
plt.show()

# Convert the trained model into dot format, save it and visualise
# the architecture of the network/model
model_saving_funcs.save_model_arch_plot(conv_lstm_model, "conv_lstm_model")


