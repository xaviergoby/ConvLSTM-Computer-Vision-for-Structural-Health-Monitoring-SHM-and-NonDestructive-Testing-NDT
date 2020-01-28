import matplotlib.pyplot as plt
import numpy as np
import itertools



class VisTool:

	def plot_single_ts_data(self, data):
		"""
		:param data: a pd.Series of a single sample of ts data
		:return:plot
		"""
		tot_data_points = len(data)
		fig, ax = plt.subplots()
		ax.plot(list(range(tot_data_points)), data.values)
		return plt.show()

	def plot_multiple_ts_data(self, data):
		tot_data_points = data.shape[0]
		fig, ax = plt.subplots()
		ax.plot(list(range(tot_data_points)), data.values)
		return plt.show()

	def _plot_confusion_matrix(self, conf_mat, class_labels,
	                          normalize=False, title='Confusion matrix',
	                          cmap=plt.cm.Blues):
		"""
		This function prints and plots the confusion matrix.
		Normalization can be applied by setting `normalize=True`.
		"""
		if normalize:
			conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
			print("Normalized confusion matrix")
		else:
			print('Confusion matrix, without normalization')

		print(conf_mat)

		plt.imshow(conf_mat, interpolation='nearest', cmap=cmap)
		plt.title(title)
		plt.colorbar()
		tick_marks = np.arange(len(class_labels))
		plt.xticks(tick_marks, class_labels, rotation=45)
		plt.yticks(tick_marks, class_labels)

		fmt = '.2f' if normalize else 'd'
		thresh = conf_mat.max() / 2.
		for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
			plt.text(j, i, format(conf_mat[i, j], fmt),
					 horizontalalignment="center",
					 color="white" if conf_mat[i, j] > thresh else "black")

		plt.tight_layout()
		plt.ylabel('True label')
		plt.xlabel('Predicted label')

	def plot_confusion_matrix(self, conf_mat, class_labels,
	                          normalize=False, title='Confusion matrix',
	                          cmap=plt.cm.Blues):
		plt.figure()
		self._plot_confusion_matrix(conf_mat, class_labels,
	                          normalize=False, title='Confusion matrix',
	                          cmap=plt.cm.Blues)
		plt.show()

if __name__ == "__main__":
	from data_tools.load_lstm_data import LSTMDataSource
	src = LSTMDataSource()
	data = src.get_ts_data_by_sample_idx(0)
	dict_data = src.get_ts_data_by_label(2)
	# VisTool().plot_single_ts_data(data)
	VisTool().plot_multiple_ts_data(dict_data)
