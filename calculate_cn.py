import matplotlib.pyplot as plt
import seaborn as sb

sb.set_style(style="whitegrid")
sb.set_color_codes()
import numpy as np
from scipy.stats import skew
import pandas as pd
import eif_old as eif_old_class
import scipy.io as sio

# read data

# data = sio.loadmat('./datasets/cardio.mat')
# data = sio.loadmat('./datasets/annthyroid.mat')
# data = sio.loadmat('./datasets/mnist.mat')
# data = sio.loadmat('./datasets/satellite.mat')
# data = sio.loadmat('./datasets/shuttle.mat')
data = sio.loadmat('./datasets/thyroid.mat')


x_data = np.array(data["X"])
y_data = np.array(data["y"])
x_y_data = np.concatenate((x_data, y_data), axis=1)
raw_datas = x_y_data
raw_datas_without_label = x_data
print(raw_datas[0])
print(raw_datas_without_label[0])

# train the forest
number_of_trees = 1000
subsample_size = int(len(raw_datas_without_label) / 2)
extensionLevel = raw_datas_without_label.shape[1] - 1
F1 = eif_old_class.iForest(raw_datas_without_label, ntrees=number_of_trees, sample_size=subsample_size,
                           ExtensionLevel=extensionLevel)
print(F1.c)
