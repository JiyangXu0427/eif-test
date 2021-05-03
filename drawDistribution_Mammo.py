import matplotlib.pyplot as plt
import seaborn as sb
sb.set_style(style="whitegrid")
sb.set_color_codes()
import numpy as np
import pandas as pd

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