import matplotlib.pyplot as plt
import seaborn as sb

sb.set_style(style="whitegrid")
sb.set_color_codes()
import numpy as np
from scipy.stats import skew
import pandas as pd
import eif_old as eif_old_class


def plot_distribution_for_statistic_combine(statistic_title: str):
    normalFrame = pd.read_excel("./plots/statistic_normal.xlsx")
    suspectFrame = pd.read_excel("./plots/statistic_suspect.xlsx")
    anormalyFrame = pd.read_excel("./plots/statistic_anormaly.xlsx")

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

    s_statistic_value_array = np.array(suspectFrame.loc[:, statistic_title])
    s_statistic_value_series = pd.value_counts(s_statistic_value_array)
    s_statistic_value_series = s_statistic_value_series.sort_index()
    # print(lengths_count)
    indexes = s_statistic_value_series.index
    indexes_list = np.array(indexes)
    # print(indexes_list)
    value_Column = np.array(s_statistic_value_series)
    # print(path_length_result_Column)
    axTotal.plot(indexes_list, value_Column, label="s_" + statistic_title)

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
    suspectFrame = pd.read_excel("./plots/statistic_suspect.xlsx")
    anormalyFrame = pd.read_excel("./plots/statistic_anormaly.xlsx")

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
    s_statistic_value_array = np.array(suspectFrame.loc[:, statistic_title])
    s_statistic_value_series = pd.value_counts(s_statistic_value_array)
    s_statistic_value_series = s_statistic_value_series.sort_index()
    # print(lengths_count)
    indexes = s_statistic_value_series.index
    indexes_list = np.array(indexes)
    # print(indexes_list)
    value_Column = np.array(s_statistic_value_series)
    # print(path_length_result_Column)
    axTotal.plot(indexes_list, value_Column, label="s_" + statistic_title)
    axTotal.set_xlabel(statistic_title)
    axTotal.set_ylabel('Count')
    axTotal.set_title("Distribution of " + statistic_title)
    axTotal.legend()
    start, end = axTotal.get_xlim()
    stepsize = (abs(start) + abs(end)) / 8
    axTotal.xaxis.set_ticks(np.arange(start, end, stepsize))
    path = "./plots/" + "Distribution of " + statistic_title + "S" +".jpg"
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


plot_distribution_for_statistic_combine("mean")
plot_distribution_for_statistic_combine("variance")
plot_distribution_for_statistic_combine("Standard_Deviation")
plot_distribution_for_statistic_combine("Skewness")
plot_distribution_for_statistic_combine("Score")

plot_distribution_for_statistic_seperate("mean")
plot_distribution_for_statistic_seperate("variance")
plot_distribution_for_statistic_seperate("Standard_Deviation")
plot_distribution_for_statistic_seperate("Skewness")
plot_distribution_for_statistic_seperate("Score")