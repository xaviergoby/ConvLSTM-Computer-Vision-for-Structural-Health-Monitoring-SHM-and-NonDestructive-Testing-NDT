import settings
import csv
import os


# C:\Users\Xavier\LSTMforSHM\data\csv_data\csv_data_test_file.csv
# csv_file_path = r"C:\Users\Xavier\LSTMforSHM\data\csv_data\csv_data_test_file.csv"
csv_file_path = r"C:\Users\Xavier\LSTMforSHM\data\csv_data\csv_data_test_file123.csv"


def read_csv_file(csv_file_path):
	csv_data_row_elements = []
	with open(csv_file_path, "r") as csv_file:
		reader = csv.reader(csv_file, delimiter=';')
		# len(reader)
		for row in reader:
			csv_data_row_elements.append(row)
			print(', '.join(row))
	return csv_data_row_elements


def read_csv_file_2_dict(csv_file_path):
	csv_data_row_elements = []
	with open(csv_file_path) as csv_file:
		csv_dict_reader = csv.DictReader(csv_file, delimiter=';')
		print(csv_dict_reader.fieldnames)
		fieldnames_keys = csv_dict_reader.fieldnames
		for row in csv_dict_reader:
			csv_data_row_elements.append(row)
			# print(row)
			# print(row["Country"])
	return csv_data_row_elements

			

# res1 = read_csv_file(csv_file_path)
res2 = read_csv_file_2_dict(csv_file_path)
# print(res1)
print(res2)