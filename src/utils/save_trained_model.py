import datetime



def save_trained_model_arch(trained_model, file_name):
	trained_model_arch_json = trained_model.to_json()
	date_str = datetime.date.today().strftime("%d-%m-%Y")
	json_file_name = "{0}_{1}.json".format(file_name, date_str)
	with open(json_file_name, "w") as json_file:
		json_file.write(trained_model_arch_json)

def save_trained_model(trained_model, file_name):
	date_str = datetime.date.today().strftime("%d-%m-%Y")
	hdf5_file_name = "{0}_{1}.h5".format(file_name, date_str)
	trained_model.save(hdf5_file_name)

