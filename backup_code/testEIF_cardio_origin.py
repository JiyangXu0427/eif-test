import matplotlib.pyplot as plt
import seaborn as sb

sb.set_style(style="whitegrid")
sb.set_color_codes()
import numpy as np
from scipy.stats import skew
import pandas as pd
import eif_old as eif_old_class

# read data
ws = pd.read_excel("./datasets/Cardio.xlsx")
raw_datas = np.array(ws)
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
Normal_Data_by_label = []
Suspect_by_label = []
Anormaly_by_label = []

for i in range(len(DataPointsResult)):
    if DataPointsResult[i]["label"] == 1:
        # 1655 normal
        Normal_Data_by_label.append(DataPointsResult[i])
    if DataPointsResult[i]["label"] == 2:
        # 295 suspect
        Suspect_by_label.append(DataPointsResult[i])
    if DataPointsResult[i]["label"] == 3:
        # 176 pathologic
        Anormaly_by_label.append(DataPointsResult[i])

# for i in range(len(DataPointsResult)):
#     if DataPointsResult[i]["score"] > 0.5:
#         True_Anomalys.append(DataPointsResult[i])
#     else:
#         True_Normal_Data.append(DataPointsResult[i])

# plot the histogram for the path length of anomaly and normal data

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

s_means = []
s_variances = []
s_std_devs = []
s_skws = []
s_score = []
if Suspect_by_label != None:
    print("Number of Suspect:" + str(len(Suspect_by_label)))

    figTotal, axTotal = plt.subplots()
    starter = 0
    for j in range(len(Suspect_by_label)):
        lengths = Suspect_by_label[j]["length"]
        mu = np.mean(lengths)
        s_means.append(mu)
        s_variances.append(np.var(lengths))
        sigma = np.std(lengths)
        s_std_devs.append(sigma)
        s_skws.append(skew(lengths))
        score = round(Suspect_by_label[j]["score"], 2)
        s_score.append(score)

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
                     label=str(score) + "-" + "S" + "-" + str(Suspect_by_label[j]["list_num"]))

        if (j + 1) % 10 == 0 or j == int(len(Suspect_by_label) - 1):
            axTotal.set_xlabel('Path Length')
            axTotal.set_ylabel('Count')
            axTotal.set_title("Path Length Distribution")
            axTotal.legend()
            start, end = axTotal.get_xlim()
            stepsize = (start + end) / 8
            axTotal.xaxis.set_ticks(np.arange(start, end, stepsize))
            path = "./plots/" + "S" + "-" + str(starter) + "-" + str(j) + ".jpg"
            figTotal.savefig(path)
            plt.close(figTotal)
            print("Suspect " + str(j) + " plot done")

            if j != int(len(Suspect_by_label) - 1):
                figTotal, axTotal = plt.subplots()
                starter = j + 1

a_means = []
a_variances = []
a_std_devs = []
a_skws = []
a_score = []
if Anormaly_by_label != None:
    print("Number of Anormaly data:" + str(len(Anormaly_by_label)))

    figTotal, axTotal = plt.subplots()
    starter = 0
    for j in range(len(Anormaly_by_label)):
        lengths = Anormaly_by_label[j]["length"]
        mu = np.mean(lengths)
        a_means.append(mu)
        a_variances.append(np.var(lengths))
        sigma = np.std(lengths)
        a_std_devs.append(sigma)
        a_skws.append(skew(lengths))
        score = round(Anormaly_by_label[j]["score"], 2)
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
                     label=str(score) + "-" + "A" + "-" + str(Anormaly_by_label[j]["list_num"]))

        if (j + 1) % 10 == 0 or j == int(len(Anormaly_by_label) - 1):
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
            print("Anormaly " + str(j) + " plot done")

            if j != int(len(Anormaly_by_label) - 1):
                figTotal, axTotal = plt.subplots()
                starter = j + 1

nomalDataFrame = pd.DataFrame(
    {"mean": n_means, "variance": n_variances, "Standard_Deviation": n_std_devs, "Skewness": n_skws, "Score": n_score})

suspectDataFrame = pd.DataFrame(
    {"mean": s_means, "variance": s_variances, "Standard_Deviation": s_std_devs, "Skewness": s_skws, "Score": s_score})

anomalyDataFrame = pd.DataFrame(
    {"mean": a_means, "variance": a_variances, "Standard_Deviation": a_std_devs, "Skewness": a_skws, "Score": a_score})

nomalDataFrame.to_excel("./plots/statistic_normal.xlsx")
suspectDataFrame.to_excel("./plots/statistic_suspect.xlsx")
anomalyDataFrame.to_excel("./plots/statistic_anormaly.xlsx")
