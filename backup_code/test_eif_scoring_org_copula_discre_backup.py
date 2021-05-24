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


def testEIF_Scoring_data(filename, number_of_trees, subsample_size, extensionLevel, dataset_type):
    # read data from file
    if dataset_type == "10BIN":
        pd_data = pd.read_csv('./datasets/' + filename + "_discretized_10BIN_withLabel.csv")
        np_x_data = np.array(pd_data.iloc[:, :-1])
        np_y_data = np.array(pd_data.iloc[:, -1])
        np_x_y_data = np.array(pd_data)
    elif dataset_type == "15BIN":
        pd_data = pd.read_csv('./datasets/' + filename + "_discretized_15BIN_withLabel.csv")
        np_x_data = np.array(pd_data.iloc[:, :-1])
        np_y_data = np.array(pd_data.iloc[:, -1])
        np_x_y_data = np.array(pd_data)
    elif dataset_type == "copula_0.0625":
        pd_data = pd.read_excel('./datasets/' + filename + "-" + str(0.0625) + "_copula_data.xlsx")
        np_x_data = np.array(pd_data.iloc[:, :-1])
        np_y_data = np.array(pd_data.iloc[:, -1])
        np_x_y_data = np.array(pd_data)
    elif dataset_type == "copula_0.25":
        pd_data = pd.read_excel('./datasets/' + filename + "-" + str(0.25) + "_copula_data.xlsx")
        np_x_data = np.array(pd_data.iloc[:, :-1])
        np_y_data = np.array(pd_data.iloc[:, -1])
        np_x_y_data = np.array(pd_data)
    elif dataset_type == "copula_1":
        pd_data = pd.read_excel('./datasets/' + filename + "-" + str(1) + "_copula_data.xlsx")
        np_x_data = np.array(pd_data.iloc[:, :-1])
        np_y_data = np.array(pd_data.iloc[:, -1])
        np_x_y_data = np.array(pd_data)
    elif dataset_type == "copula_4":
        pd_data = pd.read_excel('./datasets/' + filename + "-" + str(4) + "_copula_data.xlsx")
        np_x_data = np.array(pd_data.iloc[:, :-1])
        np_y_data = np.array(pd_data.iloc[:, -1])
        np_x_y_data = np.array(pd_data)
    elif dataset_type == "copula_16":
        pd_data = pd.read_excel('./datasets/' + filename + "-" + str(16) + "_copula_data.xlsx")
        np_x_data = np.array(pd_data.iloc[:, :-1])
        np_y_data = np.array(pd_data.iloc[:, -1])
        np_x_y_data = np.array(pd_data)
    else:
        # origin
        data = sio.loadmat('./datasets/' + filename + '.mat')
        np_x_data = np.array(data["X"])
        np_y_data = np.array(data["y"])
        np_x_y_data = np.concatenate((np_x_data, np_y_data), axis=1)

    training_data_without_label = np_x_data
    testing_data_with_label = np_x_y_data  # here we use traning data to see their score, can be replaced by test set
    # print(raw_datas[0])
    # print(training_data_without_label[0])

    if subsample_size == "half":
        subsample_size_used = int(len(training_data_without_label) / 2)
    # elif subsample_size > int(len(training_data_without_label) / 2):
    #     subsample_size_used = int(len(training_data_without_label) / 2)
    else:
        subsample_size_used = subsample_size

    if extensionLevel == "full":
        extd_level_in_filename = extensionLevel
        extensionLevel = training_data_without_label.shape[1] - 1
    else:
        extd_level_in_filename = str(extensionLevel)

    # traing the forest
    F1 = eif_old_class.iForest(training_data_without_label, ntrees=number_of_trees, sample_size=subsample_size_used,
                               ExtensionLevel=extensionLevel)

    # compute score for testing data
    DataPointsResult = F1.compute_score_with_labeled_input(X_in=testing_data_with_label)
    # print(DataPointsResult)
    print("Scoring done")

    threshold = 0.5
    anomaly_label = 1
    normal_label = 0
    TP_Count = 0
    FP_Count = 0
    TN_Count = 0
    FN_Count = 0
    prediction_result = []
    prediction_result_confusion_matrix = []
    for i in range(DataPointsResult.shape[0]):
        score = DataPointsResult.at[i, "score"]
        label = DataPointsResult.at[i, "label"]
        if score > threshold:
            prediction_result.append(anomaly_label)
            if label == anomaly_label:
                prediction_result_confusion_matrix.append("TP")
                TP_Count = TP_Count + 1
            else:
                prediction_result_confusion_matrix.append("FP")
                FP_Count = FP_Count + 1
        if score <= threshold:
            prediction_result.append(normal_label)
            if label == normal_label:
                prediction_result_confusion_matrix.append("TN")
                TN_Count = TN_Count + 1
            else:
                prediction_result_confusion_matrix.append("FN")
                FN_Count = FN_Count + 1

    DataPointsResult["Prediction"] = prediction_result
    DataPointsResult["Confusion_Matrix"] = prediction_result_confusion_matrix
    # print(type(DataPointsResult))

    eif_result_data_path = "./plots/" + filename + "_EIF_Result_Data_" + dataset_type + "-" + str(
        number_of_trees) + "-" + str(subsample_size_used) + "-" + extd_level_in_filename + ".xlsx"

    DataPointsResult.to_excel(eif_result_data_path)
    # print(DataPointsResult)
    y_test = np.array(DataPointsResult.loc[:, "label"], dtype="int32")
    y_predict = np.array(DataPointsResult.loc[:, "Prediction"], dtype="int32")
    y_score = np.array(DataPointsResult.loc[:, "score"])
    # print(y_score)

    result_summary_path = "./plots/" + filename + "_EIF_Result_Summary.txt"
    with open(result_summary_path, 'a') as opened_file:
        opened_file.write("Data_Type: " + dataset_type)
        opened_file.write("\n")
        opened_file.write("Number of Trees: " + str(number_of_trees))
        opened_file.write("\n")
        opened_file.write("Subsample Size: " + str(subsample_size_used))
        opened_file.write("\n")
        opened_file.write("Extend Level: " + extd_level_in_filename)
        opened_file.write("\n")
        opened_file.write("\n")

        report = skm.classification_report(y_test, y_predict, labels=[0, 1], target_names=["Normal", "Anomaly"])
        # print(report)
        opened_file.write(report)
        opened_file.write("\n")

        result_value = skm.roc_auc_score(y_test, y_score)
        print("ROC_AUC_Score: " + str(result_value))
        opened_file.write("ROC_AUC_Score: " + str(result_value))
        opened_file.write("\n")

        result_value = skm.accuracy_score(y_test, y_predict)
        print("Overall_Accuracy_Score: " + str(result_value))
        opened_file.write("Overall_Accuracy_Score: " + str(result_value))
        opened_file.write("\n")

        result_value = skm.balanced_accuracy_score(y_test, y_predict)
        print("Overall_Balanced_Accuracy_Score: " + str(result_value))
        opened_file.write("Overall_Balanced_Accuracy_Score: " + str(result_value))
        opened_file.write("\n")

        result_value = skm.precision_score(y_test, y_predict)
        print("Overall_Precision_Score: " + str(result_value))
        opened_file.write("Overall_Precision_Score: " + str(result_value))
        opened_file.write("\n")

        result_value = skm.average_precision_score(y_test, y_predict)
        print("Overall_Average_Precision_Score: " + str(result_value))
        opened_file.write("Overall_Average_Precision_Score: " + str(result_value))
        opened_file.write("\n")

        result_value = skm.recall_score(y_test, y_predict)
        print("Overall_Recall_Score: " + str(result_value))
        opened_file.write("Overall_Recall_Score: " + str(result_value))
        opened_file.write("\n")

        result_value = skm.f1_score(y_test, y_predict)
        print("Overall_F1_Score: " + str(result_value))
        opened_file.write("Overall_F1_Score: " + str(result_value))
        opened_file.write("\n")
        opened_file.write("\n")
        opened_file.write("\n")

        fpr, tpr, thresholds = skm.roc_curve(y_test, y_score, pos_label=1)

    return fpr, tpr, subsample_size_used

dataset_names = ["annthyroid", "cardio", "foresttype", "ionosphere","mammography" ,"satellite", "shuttle", "thyroid"]
dataset_types = ["origin", "copula_0.0625", "copula_0.25", "copula_1", "copula_4", "copula_16", "10BIN", "15BIN"]
# parameter for traing the forest
number_of_trees = 500
subsample_size = 256
extensionLevel = "full"

for dataset_name in dataset_names:
    fig, ax = plt.subplots()
    for dataset_type in dataset_types:
        fpr, tpr, subsample_size_used = testEIF_Scoring_data(filename=dataset_name, number_of_trees=number_of_trees,
                                                             subsample_size=subsample_size,
                                                             extensionLevel=extensionLevel,
                                                             dataset_type=dataset_type)
        if dataset_type == "10BIN":
            plot_color = "brown"
            plot_linestyle = "--"
        elif dataset_type == "15BIN":
            plot_color = "darkblue"
            plot_linestyle = "--"
        elif dataset_type == "copula_0.0625":
            plot_color = "dimgrey"
            plot_linestyle = "dotted"
        elif dataset_type == "copula_0.25":
            plot_color = "lime"
            plot_linestyle = "dotted"
        elif dataset_type == "copula_1":
            plot_color = "royalblue"
            plot_linestyle = "dotted"
        elif dataset_type == "copula_4":
            plot_color = "yellow"
            plot_linestyle = "dotted"
        elif dataset_type == "copula_16":
            plot_color = "red"
            plot_linestyle = "dotted"
        else:
            # origin
            plot_color = "black"
            plot_linestyle = "-"

        ax.plot(fpr, tpr, label=dataset_type,color=plot_color, linestyle=plot_linestyle)

    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title("ROC Curve")
    ax.legend()
    if extensionLevel != "full":
        extensionLevel_str = extensionLevel
    else:
        extensionLevel_str = str(extensionLevel)

    plot_path = "./plots/" + dataset_name + "-EIF-ROC_Curve-" + str(number_of_trees) + "-" + str(
        subsample_size_used) + "-" + extensionLevel_str + ".jpg"

    fig.savefig(plot_path)
    plt.close(fig)
