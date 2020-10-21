import numpy as np
import matplotlib.pyplot as plt


def sine_func(x_i):
	y = np.sin(2*np.pi*x_i)
	return y

def gen_toy_sine_curve_dataset(N, std, periods=1):
	"""
	:param N: Number of data points/samples
	:param std: Standard deviation for (white/Gaussian) noise input
	:return: 2-tuple (x_vector, y_train)
	"""
	y_vector = []
	x_vector = []
	for period_i in range(periods):
		for x_i in np.linspace(0, 1, N):
			x_i = x_i + period_i
			noise = np.random.normal(0, std)
			y_i = sine_func(x_i) + noise
			x_vector.append(x_i)
			y_vector.append(y_i)
	return x_vector, y_vector

N = 50
periods = 3
std_noise = 0

x_train, y_train = gen_toy_sine_curve_dataset(N, std_noise, periods)
# x_vector_test, y_hat_vector = gen_toy_sine_curve_dataset(10, 0.25)

fig_dims = (12, 12)
fig, ax = plt.subplots(figsize=fig_dims)
# fig, axes = plt.subplots(1, 3, figsize = fig_dims)
ax.plot(x_train, y_train, "g", label="Sine curve plot (x_train vs y_train) - num data points = {0} std = {1}".format(N, std_noise))
ax.grid(True)
ax.legend()
# plt.plot(x_train, y_train, "g")
# plt.plot(x_vector_test, y_hat_vector, "r")
plt.show()