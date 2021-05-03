import matplotlib.pyplot as plt
import seaborn as sb

sb.set_style(style="whitegrid")
sb.set_color_codes()
import numpy as np
from scipy.stats import skew
import pandas as pd
import eif_old as eif_old_class

# read data
raw_datas = np.loadtxt("./datasets/Mulcross.csv", usecols=(0, 1, 2, 3, 4), skiprows=1, delimiter=",",
                       unpack=False)
raw_datas_without_label = raw_datas[:, :-1]
# print(raw_datas[0])
# print(raw_datas_without_label[0])

# train the forest
number_of_trees = 1000
subsample_size = int(len(raw_datas_without_label) / 2)
extensionLevel = raw_datas_without_label.shape[1] - 1
F1 = eif_old_class.iForest(raw_datas_without_label, ntrees=number_of_trees, sample_size=subsample_size,
                           ExtensionLevel=extensionLevel)

# compute score for testing data
testing_data = raw_datas  # here we use traning data to see their score, can be replaced by test set
S, DataPointsResult = F1.compute_paths_with_labeled_input(X_in=testing_data)
print("computing done")

# split the result into anomaly and normal lists
# the threshold can be modified from 0 - 1
# threshold = 0.5
Anomalys_by_label = []
Normal_Data_by_label = []

for i in range(len(DataPointsResult)):
    if DataPointsResult[i]["label"] == 1:
        # class 1, 260
        Anomalys_by_label.append(DataPointsResult[i])
    else:
        # class 0, 10923
        Normal_Data_by_label.append(DataPointsResult[i])

a_means = []
a_variances = []
a_std_devs = []
a_skws = []
a_score = []
if Anomalys_by_label != None:
    print("Number of Anomaly:" + str(len(Anomalys_by_label)))

    figTotal, axTotal = plt.subplots()
    starter = 0
    for j in range(len(Anomalys_by_label)):
        lengths = Anomalys_by_label[j]["length"]
        mu = np.mean(lengths)
        a_means.append(mu)
        a_variances.append(np.var(lengths))
        sigma = np.std(lengths)
        a_std_devs.append(sigma)
        a_skws.append(skew(lengths))
        score = round(Anomalys_by_label[j]["score"], 2)
        a_score.append(score)

        lengths_count = pd.value_counts(lengths)
        # print(lengths_count)
        lengths_count = lengths_count.sort_index()
        # print(lengths_count)
        indexes = lengths_count.index
        indexes_list = np.array(indexes)
        # print(indexes_list)
        path_length_result_Column = np.array(lengths_count)
        # print(path_length_result_Column)
        axTotal.plot(indexes_list, path_length_result_Column,
                     label=str(score) + "-" + "A" + "-" + str(Anomalys_by_label[j]["list_num"]))

        if (j + 1) % 10 == 0 or j == int(len(Anomalys_by_label) - 1):
            axTotal.set_xlabel('Path Length')
            axTotal.set_ylabel('Count')
            axTotal.set_title("Path Length Distribution")
            axTotal.legend()
            start, end = axTotal.get_xlim()
            stepsize = (start + end) / 8
            axTotal.xaxis.set_ticks(np.arange(start, end, stepsize))
            path = "./plots/" + "A" + "-" + str(starter) + "-" + str(j) + ".jpg"
            figTotal.savefig(path)
            plt.close(figTotal)
            print("Anomaly " + str(j) + " plot done")

            if j != int(len(Anomalys_by_label) - 1):
                figTotal, axTotal = plt.subplots()
                starter = j + 1

n_means = []
n_variances = []
n_std_devs = []
n_skws = []
n_score = []
if Normal_Data_by_label != None:
    print("Number of normal data:" + str(len(Normal_Data_by_label)))

    figTotal, axTotal = plt.subplots()
    starter = 0
    for j in range(len(Normal_Data_by_label)):
        lengths = Normal_Data_by_label[j]["length"]
        mu = np.mean(lengths)
        n_means.append(mu)
        n_variances.append(np.var(lengths))
        sigma = np.std(lengths)
        n_std_devs.append(sigma)
        n_skws.append(skew(lengths))
        score = round(Normal_Data_by_label[j]["score"], 2)
        n_score.append(score)

        lengths_count = pd.value_counts(lengths)
        # print(lengths_count)
        lengths_count = lengths_count.sort_index()
        # print(lengths_count)
        indexes = lengths_count.index
        indexes_list = np.array(indexes)
        # print(indexes_list)
        path_length_result_Column = np.array(lengths_count)
        # print(path_length_result_Column)
        axTotal.plot(indexes_list, path_length_result_Column,
                     label=str(score) + "-" + "N" + "-" + str(Normal_Data_by_label[j]["list_num"]))

        if (j + 1) % 10 == 0 or j == int(len(Normal_Data_by_label) - 1):
            axTotal.set_xlabel('Path Length')
            axTotal.set_ylabel('Count')
            axTotal.set_title("Path Length Distribution")
            axTotal.legend()
            start, end = axTotal.get_xlim()
            stepsize = (start + end) / 8
            axTotal.xaxis.set_ticks(np.arange(start, end, stepsize))
            path = "./plots/" + "N" + "-" + str(starter) + "-" + str(j) + ".jpg"
            figTotal.savefig(path)
            plt.close(figTotal)
            print("Normal " + str(j) + " plot done")

            if j != int(len(Normal_Data_by_label) - 1):
                figTotal, axTotal = plt.subplots()
                starter = j + 1

anomalyDataFrame = pd.DataFrame(
    {"mean": a_means, "variance": a_variances, "Standard_Deviation": a_std_devs, "Skewness": a_skws, "Score": a_score})

nomalDataFrame = pd.DataFrame(
    {"mean": n_means, "variance": n_variances, "Standard_Deviation": n_std_devs, "Skewness": n_skws, "Score": n_score})

anomalyDataFrame.to_excel("./plots/statistic_anomaly.xlsx")
nomalDataFrame.to_excel("./plots/statistic_normal.xlsx")


def plot_distribution_for_statistic_combined(statistic_title: str):
    normalFrame = pd.read_excel("./plots/statistic_normal.xlsx")
    anormalyFrame = pd.read_excel("./plots/statistic_anomaly.xlsx")
    figTotal, axTotal = plt.subplots()
    n_statistic_value_array = np.array(normalFrame.loc[:, statistic_title])
    n_statistic_value_series = pd.value_counts(n_statistic_value_array)
    n_statistic_value_series = n_statistic_value_series.sort_index()
    # print(lengths_count)
    indexes = n_statistic_value_series.index
    indexes_list = np.array(indexes)
    # print(indexes_list)
    value_Column = np.array(n_statistic_value_series)
    # print(path_length_result_Column)
    axTotal.plot(indexes_list, value_Column, label="n_" + statistic_title)

    a_statistic_value_array = np.array(anormalyFrame.loc[:, statistic_title])
    a_statistic_value_series = pd.value_counts(a_statistic_value_array)
    a_statistic_value_series = a_statistic_value_series.sort_index()
    # print(lengths_count)
    indexes = a_statistic_value_series.index
    indexes_list = np.array(indexes)
    # print(indexes_list)
    value_Column = np.array(a_statistic_value_series)
    # print(path_length_result_Column)
    axTotal.plot(indexes_list, value_Column, label="a_" + statistic_title)

    axTotal.set_xlabel(statistic_title)
    axTotal.set_ylabel('Count')
    axTotal.set_title("Distribution of " + statistic_title)
    axTotal.legend()
    start, end = axTotal.get_xlim()
    stepsize = (abs(start) + abs(end)) / 8
    axTotal.xaxis.set_ticks(np.arange(start, end, stepsize))
    path = "./plots/" + "Distribution of " + statistic_title + ".jpg"
    figTotal.savefig(path)
    plt.close(figTotal)
    return


def plot_distribution_for_statistic_seperate(statistic_title: str):
    normalFrame = pd.read_excel("./plots/statistic_normal.xlsx")
    anormalyFrame = pd.read_excel("./plots/statistic_anomaly.xlsx")

    figTotal, axTotal = plt.subplots()
    n_statistic_value_array = np.array(normalFrame.loc[:, statistic_title])
    n_statistic_value_series = pd.value_counts(n_statistic_value_array)
    n_statistic_value_series = n_statistic_value_series.sort_index()
    # print(lengths_count)
    indexes = n_statistic_value_series.index
    indexes_list = np.array(indexes)
    # print(indexes_list)
    value_Column = np.array(n_statistic_value_series)
    # print(path_length_result_Column)
    axTotal.plot(indexes_list, value_Column, label="n_" + statistic_title)
    axTotal.set_xlabel(statistic_title)
    axTotal.set_ylabel('Count')
    axTotal.set_title("Distribution of " + statistic_title)
    axTotal.legend()
    start, end = axTotal.get_xlim()
    stepsize = (abs(start) + abs(end)) / 8
    axTotal.xaxis.set_ticks(np.arange(start, end, stepsize))
    path = "./plots/" + "Distribution of " + statistic_title + "N" +".jpg"
    figTotal.savefig(path)
    plt.close(figTotal)

    figTotal, axTotal = plt.subplots()
    a_statistic_value_array = np.array(anormalyFrame.loc[:, statistic_title])
    a_statistic_value_series = pd.value_counts(a_statistic_value_array)
    a_statistic_value_series = a_statistic_value_series.sort_index()
    # print(lengths_count)
    indexes = a_statistic_value_series.index
    indexes_list = np.array(indexes)
    # print(indexes_list)
    value_Column = np.array(a_statistic_value_series)
    # print(path_length_result_Column)
    axTotal.plot(indexes_list, value_Column, label="a_" + statistic_title)

    axTotal.set_xlabel(statistic_title)
    axTotal.set_ylabel('Count')
    axTotal.set_title("Distribution of " + statistic_title)
    axTotal.legend()
    start, end = axTotal.get_xlim()
    stepsize = (abs(start) + abs(end)) / 8
    axTotal.xaxis.set_ticks(np.arange(start, end, stepsize))
    path = "./plots/" + "Distribution of " + statistic_title + "A" +".jpg"
    figTotal.savefig(path)
    plt.close(figTotal)
    return

plot_distribution_for_statistic_combined("mean")
plot_distribution_for_statistic_combined("variance")
plot_distribution_for_statistic_combined("Standard_Deviation")
plot_distribution_for_statistic_combined("Skewness")
plot_distribution_for_statistic_combined("Score")

plot_distribution_for_statistic_seperate("mean")
plot_distribution_for_statistic_seperate("variance")
plot_distribution_for_statistic_seperate("Standard_Deviation")
plot_distribution_for_statistic_seperate("Skewness")
plot_distribution_for_statistic_seperate("Score")