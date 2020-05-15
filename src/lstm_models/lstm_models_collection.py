from keras.layers import GRU, LSTM, Dense, Dropout, LeakyReLU

def get_simple_single_layer_fc_lstm_func_api_model_output(model_input, num_classes):
    # LSTM: First and following return sequences must be True, Last return sequence must be False
    lstm_1 = LSTM(8, return_sequences=True, activation="relu")(model_input)
    do1 = Dropout(0.25)(lstm_1)
    lstm_2 = LSTM(16, return_sequences=True, activation="relu")(do1)
    do2 = Dropout(0.25)(lstm_2)
    lstm_3 = LSTM(32, return_sequences=True, activation="relu")(do2)
    do3 = Dropout(0.25)(lstm_3)
    lstm_4 = LSTM(16, return_sequences=False, activation="relu")(do3)
    do4 = Dropout(0.25)(lstm_4)
    #MLP
    #d2 = Dense(16, activation="relu")(do4)
    #do5 = Dropout(0.25)(d2)
    d3 = Dense(num_classes, activation="softmax")(do4)
    return d3