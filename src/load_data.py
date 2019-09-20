import pandas as pd


# data = pd.read_csv("../data/data.csv")
# labels = pd.read_csv("../data/label.csv")

class DataSource:
	
	def __init__(self, data_file_path = "../data/data.csv", labels_file_path = "../data/labels.csv"):
		self.data_file_path = data_file_path
		self.labels_file_path = labels_file_path
		self.ts_data = pd.read_csv("../data/data.csv")
		self.labels = pd.read_csv("../data/label.csv")
		
		
	def create_labels_train_ts_data_dict(self):
		labels_ts_data_dict = {}
		all_unique_labels_list = self.labels["0"].drop_duplicates().to_list()
		for label_i in all_unique_labels_list:
			label_i_ts_df = self.labels[self.labels == label_i]
			label_i_ts_df = label_i_ts_df.dropna()
			label_i_ts_list = label_i_ts_df["0"].to_list()
			labels_ts_data_dict[label_i] = label_i_ts_list
		return labels_ts_data_dict
	
	
if __name__ == "__main__":
	data_src = DataSource()
	dict = data_src.create_labels_train_ts_data_dict()
	print(dict.keys())
	print(len(dict[0]), len(dict[1]), len(dict[2]), len(dict[3]))
	
	
	import matplotlib.pyplot as plt
	xs = list(range(0, len(dict[0]) + 1))
	d1 = dict[1]
	d2 = dict[2]
	plt.plot(xs, data_src.ts_data)
	plt.plot(d2, xs)
	plt.show()
	
	