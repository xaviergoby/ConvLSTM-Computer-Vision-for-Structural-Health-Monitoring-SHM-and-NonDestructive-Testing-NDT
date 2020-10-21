import configs_and_settings
import os


def get_file_folder_names_in_dir(dir_path):
	files_folders_names_list = [file_i for file_i in os.listdir(dir_path)]
	return files_folders_names_list

def get_num_files_in_dir(dir_path):
	files_folders_names_list = get_file_folder_names_in_dir(dir_path)
	num_files_folders = len(files_folders_names_list)
	return num_files_folders