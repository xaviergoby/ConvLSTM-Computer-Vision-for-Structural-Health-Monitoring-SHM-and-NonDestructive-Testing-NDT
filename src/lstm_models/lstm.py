#
import pandas as pd
import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM, Dropout
from keras.utils import to_categorical
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt

def readData():
	# Get labels from the labels.txt file
	labels = pd.read_csv('label.csv', header = None)
	labels = labels.values
	#labels = labels-1
	print('One Hot Encoding Data...')
	labels = to_categorical(labels)

	data = pd.read_csv('data.csv', header = None)

	return data, labels

print('Reading data...')
data, labels = readData()
print(data.shape, labels.shape)
data = np.expand_dims(data, axis=-1)
print(data.shape, labels.shape)

# print('Splitting Data')
# data_train, data_test, labels_train, labels_test = train_test_split(data, labels)
#
# print('Building Model...')
# #Create model
# model = Sequential()
# model.add(LSTM(units=32, input_shape = (801,1), return_sequences=True))
# model.add(LSTM(units=64, return_sequences=True))
# model.add(LSTM(units=128))
# model.add(Dropout(0.5))
# model.add(Dense(128))
# model.add(Dense(6, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#
# print('Training NN...')
# history = model.fit(data_train, labels_train, epochs=100, batch_size=5,
# 	validation_split=0.25,verbose=1)
#
# results = model.evaluate(data_test, labels_test)
#
# predictions = model.predict(data_test)
#
# print(predictions[0].shape)
# print(np.sum(predictions[0]))
# print(np.argmax(predictions[0]))
#
# print(results)
#
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# epochs = range(1, len(acc) + 1)
#
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()