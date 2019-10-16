import pandas as pd
import matplotlib.pyplot as plt


# data = pd.read_csv("../data/data.csv")
# labels = pd.read_csv("../data/label.csv")

class DataSource:
	
	def __init__(self, data_file_path = "../data/data.csv", labels_file_path = "../data/labels.csv"):
		self.data_file_path = data_file_path
		self.labels_file_path = labels_file_path
		self.ts_data_idx_headers = list(range(1, 802))
		self.ts_data = pd.read_csv("../data/data.csv", header=None)
		self.labels = pd.read_csv("../data/label.csv", header=None, names=["Labels"])
		
		
	def create_labels_train_ts_data_dict(self):
		labels_ts_data_dict = {}
		all_unique_labels_list = self.labels["0"].drop_duplicates().to_list()
		for label_i in all_unique_labels_list:
			label_i_ts_df = self.labels[self.labels == label_i]
			label_i_ts_df = label_i_ts_df.dropna()
			label_i_ts_list = label_i_ts_df["0"].to_list()
			labels_ts_data_dict[label_i] = label_i_ts_list
		return labels_ts_data_dict

	def build_labels_train_ts_df(self):
		ts_data_copy = self.ts_data.copy()
		ts_data_copy.insert(0, "Labels", self.labels["Labels"])
		return ts_data_copy

	def plot_data(self):
		# fig, ax = plt.subplots()
		label = 0
		time_steps = list(range(self.ts_data.shape[1]))
		ts_data_and_labels_df = self.build_labels_train_ts_df()
		for idx, row in ts_data_and_labels_df.iterrows():
			if row["Labels"] == label:
				fig, ax = plt.subplots()
				# ts = ts_data_and_labels_df[ts_data_and_labels_df.columns[1:]]
				ts = row[1:]
				ts_sma = ts.rolling(window=8).mean()
				ts_sma_row_data = ts_sma[7:]
				# ax.plot(time_steps, ts)
				ax.plot(time_steps[7:], ts_sma_row_data)
				plt.show()

			# continue
		# plt.show()
		# return plt.show()




	# def create_ts_data_header_idxs(self):
	# 	ts_data_idx_headers = list(range(1, 802))
	# 	cols_num = len(self.ts_data.iloc[0,0:])


	
if __name__ == "__main__":
	src = DataSource()
	# data = src.ts_data
	# labels = src.labels
	# print(data.head())
	# pd.DataFrame()
	ts_labels = src.build_labels_train_ts_df()
	print(ts_labels)
	src.plot_data()





	

	# dict = data_src.create_labels_train_ts_data_dict()
	# print(dict.keys())
	# print(len(dict[0]), len(dict[1]), len(dict[2]), len(dict[3]))

	
	# import matplotlib.pyplot as plt
	# xs = list(range(0, len(dict[0]) + 1))
	# d1 = dict[1]
	# d2 = dict[2]
	# plt.plot(xs, data_src.ts_data)
	# plt.plot(d2, xs)
	# plt.show()
	
	