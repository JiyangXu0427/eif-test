import matplotlib.pyplot as plt
import seaborn as sb

sb.set_style(style="whitegrid")
sb.set_color_codes()
import numpy as np
import scipy.stats as sts
import pandas as pd
import eif_old as eif_old_class
import scipy.io as sio
import os


def testEIF_Method_sum_length_together(filename):
    # read data from file
    data = sio.loadmat('./datasets/' + filename + '.mat')
    x_data = np.array(data["X"])
    y_data = np.array(data["y"])
    x_y_data = np.concatenate((x_data, y_data), axis=1)

    raw_datas = x_y_data
    raw_datas_without_label = x_data
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

    Anomalys_by_label = []
    Normal_Data_by_label = []

    all_anomaly_lengths = []
    all_normal_lengths = []

    for i in range(len(DataPointsResult)):
        if DataPointsResult[i]["label"] == 1:
            # class 1, 176
            Anomalys_by_label.append(DataPointsResult[i])
        else:
            # class 0, 1831 - 176
            Normal_Data_by_label.append(DataPointsResult[i])

    if Anomalys_by_label != None:
        print("Number of Anomaly:" + str(len(Anomalys_by_label)))
        for j in range(len(Anomalys_by_label)):
            lengths = Anomalys_by_label[j]["length"]
            all_anomaly_lengths.extend(lengths)

    if Normal_Data_by_label != None:
        print("Number of normal data:" + str(len(Normal_Data_by_label)))
        for j in range(len(Normal_Data_by_label)):
            lengths = Normal_Data_by_label[j]["length"]
            all_normal_lengths.extend(lengths)

    lengths_count = pd.value_counts(all_anomaly_lengths)
    # print(lengths_count)
    lengths_count = lengths_count.sort_index()
    # print(lengths_count)
    indexes = lengths_count.index
    indexes_list = np.array(indexes)
    # print(indexes_list)
    path_length_result_Column = np.array(lengths_count)
    # print(path_length_result_Column)

    fig_ten, ax_ten = plt.subplots()
    ax_ten.plot(indexes_list, path_length_result_Column)
    ax_ten.set_xlabel('Path Length')
    ax_ten.set_ylabel('Count')
    ax_ten.set_title("Anomaly Path Length Distribution")
    start, end = ax_ten.get_xlim()
    stepsize = (start + end) / 8
    ax_ten.xaxis.set_ticks(np.arange(start, end, stepsize))
    path = "./plots/" + filename + "-" + "Anomaly_Path_length_distribution.jpg"
    fig_ten.savefig(path)
    plt.close(fig_ten)

    lengths_count = pd.value_counts(all_normal_lengths)
    # print(lengths_count)
    lengths_count = lengths_count.sort_index()
    # print(lengths_count)
    indexes = lengths_count.index
    indexes_list = np.array(indexes)
    # print(indexes_list)
    path_length_result_Column = np.array(lengths_count)
    # print(path_length_result_Column)

    fig_ten, ax_ten = plt.subplots()
    ax_ten.plot(indexes_list, path_length_result_Column)
    ax_ten.set_xlabel('Path Length')
    ax_ten.set_ylabel('Count')
    ax_ten.set_title("Normal Path Length Distribution")
    start, end = ax_ten.get_xlim()
    stepsize = (start + end) / 8
    ax_ten.xaxis.set_ticks(np.arange(start, end, stepsize))
    path = "./plots/" + filename + "-" + "Normal_Path_length_distribution.jpg"
    fig_ten.savefig(path)
    plt.close(fig_ten)


def testEIF_Method(filename):
    # read data from file
    data = sio.loadmat('./datasets/' + filename + '.mat')
    x_data = np.array(data["X"])
    y_data = np.array(data["y"])
    x_y_data = np.concatenate((x_data, y_data), axis=1)

    raw_datas = x_y_data
    raw_datas_without_label = x_data
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
            # class 1, 176
            Anomalys_by_label.append(DataPointsResult[i])
        else:
            # class 0, 1831 - 176
            Normal_Data_by_label.append(DataPointsResult[i])

    a_means = []
    a_variances = []
    a_std_devs = []
    a_skws = []
    a_kurto = []
    a_score = []

    # draw the plot, calculate the statistic
    if Anomalys_by_label != None:
        print("Number of Anomaly:" + str(len(Anomalys_by_label)))

        fig_all, ax_all = plt.subplots()
        fig_ten, ax_ten = plt.subplots()
        starter = 0
        for j in range(len(Anomalys_by_label)):
            lengths = Anomalys_by_label[j]["length"]
            mu = np.mean(lengths)
            a_means.append(mu)
            a_variances.append(np.var(lengths))
            sigma = np.std(lengths)
            a_std_devs.append(sigma)
            a_skws.append(sts.skew(lengths))
            a_kurto.append(sts.kurtosis(lengths))
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

            ax_ten.plot(indexes_list, path_length_result_Column,
                        label=str(score) + "-" + "A" + "-" + str(Anomalys_by_label[j]["list_num"]))

            if (j + 1) % 10 == 0 or j == int(len(Anomalys_by_label) - 1):
                ax_ten.set_xlabel('Path Length')
                ax_ten.set_ylabel('Count')
                ax_ten.set_title("Path Length Distribution")
                ax_ten.legend()
                start, end = ax_ten.get_xlim()
                stepsize = (start + end) / 8
                ax_ten.xaxis.set_ticks(np.arange(start, end, stepsize))
                path = "./plots/" + "A" + "-" + str(starter) + "-" + str(j) + ".jpg"
                fig_ten.savefig(path)
                plt.close(fig_ten)
                print("Anomaly " + str(j) + " plot done")

                if j != int(len(Anomalys_by_label) - 1):
                    fig_ten, ax_ten = plt.subplots()
                    starter = j + 1

            ax_all.plot(indexes_list, path_length_result_Column)
            if j == int(len(Anomalys_by_label)) - 1:
                ax_all.set_xlabel('Path Length')
                ax_all.set_ylabel('Count')
                ax_all.set_title("Path Length Distribution for all anomaly")
                start, end = ax_all.get_xlim()
                stepsize = (start + end) / 8
                ax_all.xaxis.set_ticks(np.arange(start, end, stepsize))
                path = "./plots/" + "A" + "-" + "All.jpg"
                fig_all.savefig(path)
                plt.close(fig_all)

    n_means = []
    n_variances = []
    n_std_devs = []
    n_skws = []
    n_kurto = []
    n_score = []

    if Normal_Data_by_label != None:
        print("Number of normal data:" + str(len(Normal_Data_by_label)))

        fig_all, ax_all = plt.subplots()
        fig_ten, ax_ten = plt.subplots()
        starter = 0
        for j in range(len(Normal_Data_by_label)):
            lengths = Normal_Data_by_label[j]["length"]
            mu = np.mean(lengths)
            n_means.append(mu)
            n_variances.append(np.var(lengths))
            sigma = np.std(lengths)
            n_std_devs.append(sigma)
            n_skws.append(sts.skew(lengths))
            n_kurto.append(sts.kurtosis(lengths))
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
            ax_ten.plot(indexes_list, path_length_result_Column,
                        label=str(score) + "-" + "N" + "-" + str(Normal_Data_by_label[j]["list_num"]))

            if (j + 1) % 10 == 0 or j == int(len(Normal_Data_by_label) - 1):
                ax_ten.set_xlabel('Path Length')
                ax_ten.set_ylabel('Count')
                ax_ten.set_title("Path Length Distribution")
                ax_ten.legend()
                start, end = ax_ten.get_xlim()
                stepsize = (start + end) / 8
                ax_ten.xaxis.set_ticks(np.arange(start, end, stepsize))
                path = "./plots/" + "N" + "-" + str(starter) + "-" + str(j) + ".jpg"
                fig_ten.savefig(path)
                plt.close(fig_ten)
                print("Normal " + str(j) + " plot done")

                if j != int(len(Normal_Data_by_label) - 1):
                    fig_ten, ax_ten = plt.subplots()
                    starter = j + 1

            ax_all.plot(indexes_list, path_length_result_Column)
            if j == int(len(Normal_Data_by_label)) - 1:
                ax_all.set_xlabel('Path Length')
                ax_all.set_ylabel('Count')
                ax_all.set_title("Path Length Distribution for all normal")
                start, end = ax_all.get_xlim()
                stepsize = (start + end) / 8
                ax_all.xaxis.set_ticks(np.arange(start, end, stepsize))
                path = "./plots/" + "N" + "-" + "All.jpg"
                fig_all.savefig(path)
                plt.close(fig_all)

    anomaly_label_array = np.ones(len(Anomalys_by_label), dtype=int)
    normal_label_array = np.zeros(len(Normal_Data_by_label), dtype=int)

    cn = F1.c
    a_mean_normalised = np.array(a_means) / cn
    a_stddev_normalised = np.array(a_std_devs) / cn
    a_variance_normalised = np.array(a_variances) / cn
    n_mean_normalised = np.array(n_means) / cn
    n_stddev_normalised = np.array(n_std_devs) / cn
    n_variance_normalised = np.array(n_variances) / cn

    anomalyDataFrame = pd.DataFrame(
        {"mean": a_means, "variance": a_variances,
         "Standard_Deviation": a_std_devs, "Skewness": a_skws, "Kurtosis": a_kurto, "Score": a_score,
         "label": anomaly_label_array,
         "mean_normalised": a_mean_normalised, "Standard_Deviation_normalised": a_stddev_normalised,
         "variance_normalised": a_variance_normalised
         })

    nomalDataFrame = pd.DataFrame(
        {"mean": n_means, "variance": n_variances,
         "Standard_Deviation": n_std_devs, "Skewness": n_skws, "Kurtosis": n_kurto, "Score": n_score,
         "label": normal_label_array,
         "mean_normalised": n_mean_normalised, "Standard_Deviation_normalised": n_stddev_normalised,
         "variance_normalised": n_variance_normalised
         })

    statistic_data = pd.concat([anomalyDataFrame, nomalDataFrame], axis=0)

    anomalyDataFrame.to_excel("./plots/" + filename + "_statistic_anomaly.xlsx")
    nomalDataFrame.to_excel("./plots/" + filename + "_statistic_normal.xlsx")
    statistic_data.to_excel("./plots/" + filename + "_statistic_data.xlsx")


def plot_distribution_for_statistic_combined(statistic_title: str, filename: str):
    normalFrame = pd.read_excel("./plots/" + filename + "_statistic_normal.xlsx")
    anormalyFrame = pd.read_excel("./plots/" + filename + "_statistic_anomaly.xlsx")
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


def plot_distribution_for_statistic_seperate(statistic_title: str, filename: str):
    normalFrame = pd.read_excel("./plots/" + filename + "_statistic_normal.xlsx")
    anormalyFrame = pd.read_excel("./plots/" + filename + "_statistic_anomaly.xlsx")

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
    path = "./plots/" + "Distribution of " + statistic_title + "N" + ".jpg"
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
    path = "./plots/" + "Distribution of " + statistic_title + "A" + ".jpg"
    figTotal.savefig(path)
    plt.close(figTotal)
    return


def mk_new_dir(path: str):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print
        "---  new folder...  ---"
        print
        "---  OK  ---"
    else:
        print
        "---  There is this folder!  ---"

# file_name_input = "annthyroid"
# file_name_input = "cardio"
# file_name_input = "mnist"
# file_name_input = "satellite"
# file_name_input = "shuttle"
file_name_input = "thyroid"


testEIF_Method_sum_length_together(file_name_input)




# testEIF_Method(file_name_input)

# plot_distribution_for_statistic_combined("mean", file_name_input)
# plot_distribution_for_statistic_combined("variance", file_name_input)
# plot_distribution_for_statistic_combined("Standard_Deviation", file_name_input)
# plot_distribution_for_statistic_combined("Skewness", file_name_input)
# plot_distribution_for_statistic_combined("Score", file_name_input)
# plot_distribution_for_statistic_combined("Kurtosis", file_name_input)

# plot_distribution_for_statistic_seperate("mean", file_name_input)
# plot_distribution_for_statistic_seperate("variance", file_name_input)
# plot_distribution_for_statistic_seperate("Standard_Deviation", file_name_input)
# plot_distribution_for_statistic_seperate("Skewness", file_name_input)
# plot_distribution_for_statistic_seperate("Score", file_name_input)
# plot_distribution_for_statistic_seperate("Kurtosis", file_name_input)
