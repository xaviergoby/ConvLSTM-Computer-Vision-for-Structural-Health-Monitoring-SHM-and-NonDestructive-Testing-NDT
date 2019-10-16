

sensors_tot_num = 8

for sensor_num_i in range(1, sensors_tot_num + 1):
    train_data_dir = "multiple/3200/sensor{0}/training".format(sensor_num_i)
    print("Loading data from {0}".format(train_data_dir))
