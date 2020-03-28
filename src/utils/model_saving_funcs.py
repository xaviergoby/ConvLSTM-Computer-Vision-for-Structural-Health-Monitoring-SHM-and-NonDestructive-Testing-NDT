import datetime
from keras.utils import plot_model
import settings
import os



def save_trained_model_json_arch(trained_model, file_name):
	trained_model_arch_json = trained_model.to_json()
	date_str = datetime.date.today().strftime("%d-%m-%Y")
	json_file_name = "{0}_{1}.json".format(file_name, date_str)
	json_file_full_path = os.path.join(settings.RESULTS_DIR, r"trained_models\{0}".format(json_file_name))
	with open(json_file_full_path, "w") as json_file:
		json_file.write(trained_model_arch_json)


def save_trained_model(trained_model, file_name):
	date_str = datetime.date.today().strftime("%d-%m-%Y")
	hdf5_file_name = "{0}_{1}.h5".format(file_name, date_str)
	hdf5_file_full_path = os.path.join(settings.RESULTS_DIR, r"trained_models\{0}".format(hdf5_file_name))
	trained_model.save(hdf5_file_full_path)


def save_model_arch_plot(model, file_name):
	date_str = datetime.date.today().strftime("%d-%m-%Y")
	arch_plot_file_name = "{0}_{1}.png".format(file_name, date_str)
	arch_plot_file_full_path = os.path.join(settings.RESULTS_DIR, r"figures\{0}".format(arch_plot_file_name))
	plot_model(model, to_file=arch_plot_file_full_path, show_shapes=True, expand_nested=True, show_layer_names=True)

