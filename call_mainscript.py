# -*- coding: utf-8 -*-
"""
Created on Sat May 16 13:45:50 2020

@author: vewald
"""

# Warning: it will take about 30-60 minutes per model depneding on the number of samples.
# 4 Frame width variations x 2 dataset (RGB, greyscale) x 4 scenarios x 1 sensor network means 32 models
# For greyscale: data source from 1 sensor, for RGB: data source from combined sensor network of 3 sensors
# 32 models x 60 minutes would mean 16 hours.

from main_v2 import mainscript
import numpy as np

# In keras, fit() is much similar to sklearn's fit method, where you pass array of features as x values and target as y_train values.
# You pass your whole dataset at once in fit method. Also, use it if you can load whole data into your memory (small dataset).
# 1660 Ti GPU Memory compatible batch sizes bs = 1, 2, 4, 8, 16, etc
        
bs = 16

frame_width = np.array([128, 257, 512, 1024])
frame_height = 193
channels = 1
img_colour_format = "gray_scale"
data_dir_name = ["scene1_grey_sensor1", "scene2_grey_sensor1", "scene3_grey_sensor1", "scene4_grey_noise"]
num_epochs = 500

for ddr in data_dir_name:
    for fw in frame_width:
        mainscript.training(ddr, fw, frame_height, channels, img_colour_format, bs, num_epochs)

frame_width = np.array([128, 256, 512, 1024])
frame_height = 247
channels = 3
img_colour_format = "rgb"
data_dir_name = ["scene1_rgb_sensor123", "scene2_rgb_sensor123", "scene3_rgb_sensor123", "scene4_rgb_noise"]
num_epochs = 500

for ddr in data_dir_name:
    for fw in frame_width:
        mainscript.training(ddr, fw, frame_height, channels, img_colour_format, bs, num_epochs)