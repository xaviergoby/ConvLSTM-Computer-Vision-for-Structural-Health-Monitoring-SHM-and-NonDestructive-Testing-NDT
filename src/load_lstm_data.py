import pandas as pd
import matplotlib.pyplot as plt


# img_path = "data/SPIE2019/All/test/Sample2.png

class LSTMDataSource:
	
	def __init__(self, data_file_path = "../data/time_series/data.csv",
	             labels_file_path = "../data/time_series/label.csv"):

		self.full_data_file_path = data_file_path
		self.full_labels_file_path = labels_file_path
		# self.data = pd.read_csv("../data/data.csv", header=None)
		# self.labels = pd.read_csv("../data/label.csv", header=None, names=["Labels"])

	def get_ts_data(self):
		data = pd.read_csv(self.full_data_file_path, header=None)
		return data

	def get_labels(self):
		labels = pd.read_csv(self.full_labels_file_path, header=None, names=["Labels"])
		return labels

	def build_labels_train_ts_df(self):
		ts_data_copy = self.data.copy()
		ts_data_copy.insert(0, "Labels", self.labels["Labels"])
		return ts_data_copy

	def create_labels_train_ts_data_dict(self):
		labels = self.get_labels()
		data = self.get_ts_data()
		labelled_data_df = self.build_labels_train_ts_df()
		labels_data_dict = {}
		all_unique_labels_list = labels["Labels"].unique().tolist()
		for label_i in all_unique_labels_list:
			label_i_ts_df = labelled_data_df[labelled_data_df["Labels"] == label_i].iloc[:,1:]
			labels_data_dict[label_i] = label_i_ts_df
		return labels_data_dict

	def get_ts_data_by_label(self, label):
		ts_data_df_of_label = self.create_labels_train_ts_data_dict()[label]
		return ts_data_df_of_label

	def get_ts_data_by_sample_idx(self, sample_idx):
		data = self.get_ts_data()
		ts_sample = data.loc[sample_idx]
		return ts_sample

	
if __name__ == "__main__":
	src = LSTMDataSource()
	ts_labels = src.build_labels_train_ts_df()
	print(ts_labels)
	print(src.data.head())
	print(src.data.shape)
	print(src.labels.head())
	print(src.labels.shape)

	dict = src.create_labels_train_ts_data_dict()
	print(dict.keys())
	print(len(dict[0]), len(dict[1]), len(dict[2]), len(dict[3]), len(dict[4]), len(dict[3]))

	import matplotlib.pyplot as plt
	xs = list(range(0, len(dict[0])))
	d2 = dict[2]
	plt.plot(xs, d2)
	plt.show()
	
	