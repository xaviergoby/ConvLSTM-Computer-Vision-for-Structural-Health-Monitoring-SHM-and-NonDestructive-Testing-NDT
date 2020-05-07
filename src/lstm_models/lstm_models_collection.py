from keras.layers import TimeDistributed
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model, Sequential
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import settings
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense, LeakyReLU



def get_simple_single_layer_fc_lstm_func_api_model_output(model_input, num_classes):
    # LSTM
    lstm = LSTM(16, return_sequences=False, activation="relu")(model_input)
	# MLP
	# d1 = Dense(128, activation="relu")(lstm)
    d1 = Dense(32, activation="relu")(lstm)
    do4 = Dropout(0.25)(d1)
	# d2 = Dense(16, activation="relu")(do4)
    d2 = Dense(16, activation="relu")(do4)
    do5 = Dropout(0.25)(d2)
    d3 = Dense(num_classes, activation="softmax")(do5)
    return d3