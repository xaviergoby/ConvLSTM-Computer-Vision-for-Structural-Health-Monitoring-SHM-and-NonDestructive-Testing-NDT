# -*- coding: utf-8 -*-
"""
Created on Sat May 16 13:45:50 2020

@author: vewald
"""

from main_v2 import mainscript
import numpy as np

bs = 32
frame_width = np.array([128, 256, 512, 1024])

frame_height = 247
channels = 3
img_colour_format = "rgb"
data_dir_name = ["scene1_rgb_sensor123", "scene2_rgb_sensor123", "scene3_rgb_sensor123", "scene4_rgb_noise"]
num_epochs = 100

for ddr in data_dir_name:
    for fw in frame_width:
        mainscript.training(ddr, fw, frame_height, channels, img_colour_format, bs, num_epochs)

frame_height = 193
channels = 1
img_colour_format = "gray_scale"
data_dir_name = ["scene1_grey_sensor1", "scene2_grey_sensor1", "scene3_grey_sensor1", "scene4_grey_noise"]
num_epochs = 500

for ddr in data_dir_name:
    for fw in frame_width:
        mainscript.training(ddr, fw, frame_height, channels, img_colour_format, bs, num_epochs)