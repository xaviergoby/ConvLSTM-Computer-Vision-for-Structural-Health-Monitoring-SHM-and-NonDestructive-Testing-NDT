from keras.layers import GRU, LSTM, Dense, Dropout, LeakyReLU

do_rate = 0.25

def get_simple_single_layer_fc_lstm_func_api_model_output(model_input, num_classes):
    # LSTM: First and following return sequences must be True, Last return sequence must be False

    lstm_1 = GRU(16, return_sequences=True, activation="relu")(model_input)
    do1 = Dropout(do_rate)(lstm_1)
    lstm_2 = GRU(16, return_sequences=False, activation="relu")(do1)
    do2 = Dropout(do_rate)(lstm_2)
    #MLP
    #dense_1 = Dense(16, activation="relu")(do2)
    #do3 = Dropout(do_rate)(dense_1)
    dense_2 = Dense(num_classes, activation="softmax")(do2)
    return dense_2