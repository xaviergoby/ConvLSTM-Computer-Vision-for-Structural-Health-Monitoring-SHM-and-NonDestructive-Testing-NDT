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

num_fit_loop_iters = 10
eps = 1
bs = 1
lstm_input_layer_num_units = 20
input_tensor_shape = (1, 3)
dense_layer_num_units = 3
act_func = "softmax"
loss_metric = "mean_squared_error"
opt = "adam"


array_2_pred_1 = np.array([[a], [b]])
array_2_pred_2 = np.array([[b], [c]])
array_2_pred_3 = np.array([[c], [b]])

# training_method_desc_info = f"""The network is being trained via the combination of the .fit(.)
# 								method and a for loop. The for loop performs {}"""

# def run_with_batch_size(batch_size=bs, epochs=eps, num_fit_loop_iters=num_fit_iters):
# def run_with_batch_size(batch_size=bs, epochs=eps, num_fit_loop_iters=num_fit_iters):
def run_with_batch_size(bs, eps, num_fit_loop_iters):
	print("~" * 10 + "\tLSTM Network Creation: Start\t" + "~" * 10)
	model = Sequential()
	model.add(LSTM(lstm_input_layer_num_units, input_shape=input_tensor_shape))
	model.add(Dense(dense_layer_num_units, activation=act_func))
	model.compile(loss=loss_metric, optimizer=opt)
	print(model.summary())
	print(f"""\nLSTM Network Parameters (& Hypterparameters?):
		Batch size: {bs}
		Epochs: {eps}
		LSTM input layer number of units (dim of output space): {lstm_input_layer_num_units}
		Input tensor shape: {input_tensor_shape}
		Dense (final) layer number of units (dim of output space): {dense_layer_num_units}
		Dense (final) layer activation function: {act_func}
		Optimizer: {loss_metric}
		Loss metric: {opt}""")

	print(f"""\nData Properties:
		A single sequence of data: {seq}
		Window size: {window_size}
		Shape of training input data, x: {x.shape}
		Shape of training output data, y: {y.shape}""")
	print("~" * 10 + "\tLSTM Network Creation: End\t" + "~" * 10)

	print("\n")

	print("~"*10+"\tTraining Phase: Start\t"+"~"*10)

	# print(f"Training methodology detailed description:\n\t{training_method_desc_info}")
	print("Starting model.fit(.) & for loop training...")
	for i in range(num_fit_loop_iters):
		print("Fit Iteration Loop Number #: {0} out of {1}".format(i+1, num_fit_loop_iters))
		model.fit(x, y,
		          batch_size=bs,
		          epochs=eps,
		          verbose=2,
		          shuffle=False
		          )

	print("Ending of model.fit(.) & for loop training...")
	print("~" * 10 + "\tTraining Phase: End\t" + "~" * 10)

	print("\n")

	print("~" * 10 + "\tPrediction Phase: Start\t" + "~" * 10)
	print("~" * 5 + "\tPrediction # 1\t" + "~" * 5)
	print(f"Predicting the array:\n\t{array_2_pred_1}")
	print(f"Prediction result:\n\t{model.predict(array_2_pred_1, bs)}")
	print("\n")
	print("~" * 5 + "\tPrediction # 2\t" + "~" * 5)
	print(f"Predicting the array:\n\t{array_2_pred_2}")
	print(f"Prediction result:\n\t{model.predict(array_2_pred_2, bs)}")
	print("\n")
	print("~" * 5 + "\tPrediction # 3\t" + "~" * 5)
	print(f"Predicting the array:\n\t{array_2_pred_3}")
	print(f"Prediction result:\n\t{model.predict(array_2_pred_3, bs)}")

	print("~" * 10 + "\tPrediction Phase: End\t" + "~" * 10)



# print(f"""\nLSTM Network Parameters (& Hypterparameters?):
# 	Batch size: {bs}
# 	Epochs: {eps}
# 	LSTM input layer number of units (dim of output space): {lstm_input_layer_num_units}
# 	Input tensor shape: {input_tensor_shape}
# 	Dense (final) layer number of units (dim of output space): {dense_layer_num_units}
# 	Dense (final) layer activation function: {act_func}
# 	Optimizer: {loss_metric}
# 	Loss metric: {opt}""")
#
# print(f"""\nData Properties:
# 			\tA single sequence of data: {seq}
# 			\tWindow size: {window_size}
# 			\tTraining input and output (equivalentX_train &y_train) data properties:
# 			\t\tShape of training input data, x: {x.shape}
# 			\t\tShape of training output data, y: {y.shape}""")

print("\nStarting LSTM training and prediction demo...")
run_with_batch_size(bs, eps, num_fit_loop_iters)
print("\nLSTM training and prediction demo complete!")


def print_additional_info():
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


# print('-'*30)
# run_with_batch_size(1)
# print('**')
# run_with_batch_size(2)
