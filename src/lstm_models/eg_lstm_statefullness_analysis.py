from keras.models import Sequential
from keras.layers import Dense, LSTM
import keras.backend as K
import numpy as np
import tensorflow as tf

a = [1, 0, 0]
b = [0, 1, 0]
c = [0, 0, 1]

seq = [a, b, c, b, a]

x = seq[:-1]
y = seq[1:]
window_size = 1

x = np.array(x).reshape((len(x), window_size, 3))
y = np.array(y)


bs = 1
lstm_input_layer_num_units = 20
input_tensor_shape = (None, 3)
dense_layer_num_units = 3
act_func = "softmax"
loss_metric = "mean_squared_error"
opt = "adam"


array_2_pred_1 = np.array([[a], [b]])
array_2_pred_2 = np.array([[b], [c]])
array_2_pred_3 = np.array([[c], [b]])


def run_with_batch_size(batch_size=bs):
	model = Sequential()
	model.add(LSTM(lstm_input_layer_num_units, input_shape=input_tensor_shape))
	model.add(Dense(dense_layer_num_units, activation=act_func))
	model.compile(loss=loss_metric, optimizer=opt)

	print("~" * 10 + "\tTraining Phase:\t" + "~" * 10)
	for i in range(10):
		print("Fit Iteration AKA Epoch Number #: {0} out of {1} Epochs".format(i, 10))
		model.fit(x, y,
		          batch_size=batch_size,
		          epochs=1,
		          verbose=0,
		          shuffle=False
		          )

	print(model.predict(np.array([[a], [b]]), batch_size=batch_size))
	print()
	print(model.predict(np.array([[b], [c]]), batch_size=batch_size))
	print()
	print(model.predict(np.array([[c], [b]]), batch_size=batch_size))


print('-' * 30)
run_with_batch_size(1)
print('**')
run_with_batch_size(2)



print("\nAdditional notes on findings from the dude who asked the question:")
findings_from_author = """I have an alphabet consisting of a, b, and c
I have a sequence which is a,b,c,b,a
In this sequence a and c are always followed by b, and b is followed by c (if b was preceded by a) or by a (if b was preceded by c)
When my batch_size is 1:

I would expect that running model.predict([a, b]) gives the same result for b as when I run model.predict([b, c]) as the state was reset to zero between the two batches
The results I get match these expectations
When my batch_size is 2:

I would expect that when running model.predict([a, b]) the result for b should be affected by the result for a (since the output of a will be fed into the input of b). This means that it should have a different result for b then when I run model.predict([b, c])
The result I get is that actually the two b outputs are identical, implying that within my batch (a, b) the hidden state was reset between my two samples.
I'm still pretty fresh to this area, so it's very possible that I'm misunderstanding something. Is the initial state reset between each sample in a batch rather than between each batch?"""
print(findings_from_author)
print("Source: https://stackoverflow.com/questions/54187504/initial-states-in-keras-lstm")
