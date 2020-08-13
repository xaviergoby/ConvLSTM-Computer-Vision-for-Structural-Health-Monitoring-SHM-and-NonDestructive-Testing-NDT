import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #Activate this line to use CPU only
os.environ["PATH"] += os.pathsep + 'F:/WinPython64/python-3.7.7.amd64/Lib/graphviz-2.38/bin/' #Set path environment for graphviz library

#Importing Keras, TF Backend, and functions from /src
import tensorflow as tf
import keras
from keras.layers import TimeDistributed
from keras.layers import Input
from keras.models import Model
#from src.data_tools.image_data import image_data_handler
from matplotlib import pyplot as plt
from src.data_handling_tools.image_data_tools import image_data_handler
import matplotlib.pyplot as plt
from src.cnn_models import cnn_models_collection
from src.lstm_models import lstm_models_collection
from src.utils import model_saving_funcs
from src.utils import data_preprocessing
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

class mainscript:
    
    def training(data_dir_name, frame_width, frame_height, channels, img_colour_format, bs, num_epochs):
        
        train_dataset_dir_name = "training"
        val_dataset_dir_name = "validation"
        test_dataset_dir_name = "test"
        img_data_src = image_data_handler.ImageDataHandler(data_dir_name)
                
        # Retrieve original/raw images & class labels of training, validation and test datasets
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
        
        # CNN 2D Seq Model, both frame_height and width does not describe the actual
        # input size, but rather the amount of memory block that needs to be reserved in the RAM
        # frame_height, frame_width = None will accept arbitrary input size and would consequently
        # require "arbitrarily large" RAM. Therefore, be wise!
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
        print(conv_lstm_model.summary())
        
        # Trainer compilation, for sequential modelling RMS Prop is recommended, but Adam and SGD can be used
        lr = 0.001
        adam_optimizer = keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
        sgd_optimizer = keras.optimizers.SGD(learning_rate=lr, decay=1e-6, momentum=0.9, nesterov=True)
        rms_optimizer = keras.optimizers.RMSprop(learning_rate=lr, rho=0.9)
        
        conv_lstm_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=rms_optimizer)
        
        # Train the model. To print the training history, set verbose to 1, otherwise 0 to hide it
        history = conv_lstm_model.fit(X_train, y_train, epochs=num_epochs, verbose=0, batch_size=bs, validation_data=(X_val, y_val))
        
        # Visualising the performance by plotting the training history
        fig, ax1 = plt.subplots(figsize=(12, 8))
        color1 = 'tab:blue'
        color2 = 'tab:red'
        
        ax1.set_xlabel('Epoch',size=32)
        ax1.xaxis.set_label_coords(0.9, 1.15)
        ax1.xaxis.tick_top()
        ax1.set_ylabel('Accuracy', color=color1, size=32)
        ax1.plot(history.history['accuracy'], color=color1)
        ax1.plot(history.history['val_accuracy'], color=color1, linestyle='dashed')
        ax1.legend(['train_ACC', 'val_ACC'], loc='center right',fontsize=20, bbox_to_anchor=(0.4, -0.1),ncol = 2)
        ax1.tick_params(axis='x', labelsize = 20)
        ax1.tick_params(axis='y', labelcolor=color1, labelsize = 20)
        ax1.set_ylim([0, 1.05])
        
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('Loss', color=color2, size=32)
        ax2.plot(history.history['loss'], color=color2)
        ax2.plot(history.history['val_loss'], color=color2, linestyle='dashed')
        ax2.legend(['train_LOSS', 'val_LOSS'], loc='center right',fontsize=20, bbox_to_anchor=(1.1, -0.1),ncol = 2)
        ax2.tick_params(axis='y', labelcolor=color2, labelsize = 20)
        ax2.set_ylim([0, 3])
        
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        fig.suptitle(data_dir_name + "_" + str(frame_width), fontsize = 20, color = '0.5', x = 0.2, y = 0.97)
        fig.savefig(data_dir_name + "_" + str(frame_width))
        plt.close()
        
        # Convert the trained model into dot format, save it and visualise
        # the architecture of the network/model
        model_saving_funcs.save_model_arch_plot(conv_lstm_model, "conv_lstm_model")