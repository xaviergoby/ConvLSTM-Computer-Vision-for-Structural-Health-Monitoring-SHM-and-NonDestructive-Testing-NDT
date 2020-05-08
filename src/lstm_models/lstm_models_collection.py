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
    lstm_1 = LSTM(32, return_sequences=True, activation="relu")(model_input)
    do1 = Dropout(0.25)(lstm_1)
    lstm_2 = LSTM(16, return_sequences=False, activation="relu")(do1)
    do2 = Dropout(0.25)(lstm_2)
    #MLP
    d1 = Dense(32, activation="relu")(do2)
    do3 = Dropout(0.25)(d1)
    d2 = Dense(16, activation="relu")(do3)
    do4 = Dropout(0.25)(d2)
    d3 = Dense(num_classes, activation="softmax")(do4)
    return d3