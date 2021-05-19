import matplotlib.pyplot as plt
import seaborn as sb

sb.set_style(style="whitegrid")
sb.set_color_codes()
import numpy as np
import scipy.stats as sts
import pandas
import pandas as pd
import eif_old as eif_old_class
import scipy.io as sio
import os
import sklearn.metrics as skm


filename = "thyroid"

data = pd.read_excel('./plots/' + filename + "_copula_data.xlsx", index_col=0)
x_data = np.array(data.iloc[:, :-1])
x_y_data = np.array(data)


# data = sio.loadmat('./datasets/' + filename + '.mat')
# x_data = np.array(data["X"])
# y_data = np.array(data["y"])
# x_y_data = np.concatenate((x_data, y_data), axis=1)

for i in range(x_data.shape[1]):
    column_values = x_data[:,i]

    lengths_count = pd.value_counts(column_values)
    # print(lengths_count)
    lengths_count = lengths_count.sort_index()
    # print(lengths_count)
    indexes = lengths_count.index
    indexes_list = np.array(indexes)
    # print(indexes_list)
    path_length_result_Column = np.array(lengths_count)

    fig, ax = plt.subplots()
    ax.plot(indexes_list, path_length_result_Column, label="V"+str(i))

    ax.set_xlabel('value')
    ax.set_ylabel('count')

    ax.legend()
    plot_path = "./plots/" + filename + "_V" + str(i) + "-" + "copula"+ ".jpg"
    fig.savefig(plot_path)
    plt.close(fig)


