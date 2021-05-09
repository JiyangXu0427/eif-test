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


def testEIF_Scoring_data(filename, Copula=False):

    # read data from file
    if Copula == False:
        data = sio.loadmat('./datasets/' + filename + '.mat')
        x_data = np.array(data["X"])
        y_data = np.array(data["y"])
        x_y_data = np.concatenate((x_data, y_data), axis=1)
    else:
        data = pd.read_excel('./plots/' + filename + "_copula_data.xlsx", index_col=0)
        x_data = np.array(data.iloc[:, :-1])
        x_y_data = np.array(data)

    raw_datas = x_y_data
    raw_datas_without_label = x_data
    # print(raw_datas[0])
    # print(raw_datas_without_label[0])

    #parameter for traing the forest
    number_of_trees = 1000
    subsample_size = 1024
    if subsample_size > int(len(raw_datas_without_label) / 2) :
        subsample_size = int(len(raw_datas_without_label) / 2)
    # subsample_size = int(len(raw_datas_without_label) / 2)

    # traing the forest
    extensionLevel = raw_datas_without_label.shape[1] - 1
    F1 = eif_old_class.iForest(raw_datas_without_label, ntrees=number_of_trees, sample_size=subsample_size,
                               ExtensionLevel=extensionLevel)

    # compute score for testing data
    testing_data = raw_datas  # here we use traning data to see their score, can be replaced by test set
    DataPointsResult = F1.compute_score_with_labeled_input(X_in=testing_data)
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

    if Copula == False:
        result_data_path = "./plots/" + filename + "_Classification_Result_Data_Org-"+ str(number_of_trees) + "-" + str(subsample_size) +".xlsx"
    else:
        result_data_path = "./plots/" + filename + "_Classification_Result_Data_Copula-"+ str(number_of_trees) + "-" + str(subsample_size) +".xlsx"

    DataPointsResult.to_excel(result_data_path)
    # print(DataPointsResult)
    y_test = np.array(DataPointsResult.loc[:, "label"], dtype="int32")
    y_predict = np.array(DataPointsResult.loc[:, "Prediction"], dtype="int32")
    y_score = np.array(DataPointsResult.loc[:, "score"])
    # print(y_score)

    result_summary_path = "./plots/" + filename + "_Classification_Result_Summary.txt"
    with open(result_summary_path, 'a') as opened_file:

        if Copula == False:
            opened_file.write("\n")
            opened_file.write("Number of Trees: " + str(number_of_trees))
            opened_file.write("\n")
            opened_file.write("Subsample Size: " + str(subsample_size))
            opened_file.write("\n")
            opened_file.write("\n")
            opened_file.write("Below are original data result")
        else:
            opened_file.write("Below are Copula data result")
        opened_file.write("\n")
        opened_file.write("\n")

        report = skm.classification_report(y_test, y_predict, labels=[0, 1], target_names=["Normal", "Anomaly"])
        print(report)
        opened_file.write(report)
        opened_file.write("\n")

        result_value = skm.roc_auc_score(y_test, y_predict)
        print("ROC_AUC_Score: " + str(result_value))
        opened_file.write("ROC_AUC_Score: " + str(result_value))
        opened_file.write("\n")

        result_value = skm.accuracy_score(y_test, y_predict)
        print("Accuracy_Score: " + str(result_value))
        opened_file.write("Accuracy_Score: " + str(result_value))
        opened_file.write("\n")

        result_value = skm.balanced_accuracy_score(y_test, y_predict)
        print("Balanced_Accuracy_Score: " + str(result_value))
        opened_file.write("Balanced_Accuracy_Score: " + str(result_value))
        opened_file.write("\n")

        result_value = skm.precision_score(y_test, y_predict)
        print("Precision Score: " + str(result_value))
        opened_file.write("Precision Score: " + str(result_value))
        opened_file.write("\n")

        result_value = skm.average_precision_score(y_test, y_predict)
        print("Average Precision Score: " + str(result_value))
        opened_file.write("Average Precision Score: " + str(result_value))
        opened_file.write("\n")

        result_value = skm.recall_score(y_test, y_predict)
        print("Recall Score: " + str(result_value))
        opened_file.write("Recall Score: " + str(result_value))
        opened_file.write("\n")

        result_value = skm.f1_score(y_test, y_predict)
        print("F1 Score: " + str(result_value))
        opened_file.write("F1 Score: " + str(result_value))
        opened_file.write("\n")
        opened_file.write("\n")

        # # Accuracy = (true positive + true negative)/(All positive + All Negative)
        # Accuracy = (TP_Count+TN_Count)/(TP_Count+TN_Count+FN_Count+FP_Count)
        # print("Overall Accuracy: " + str(Accuracy))
        # opened_file.write(result_value)
        # opened_file.write("\n")
        #
        # # Precision = (true positive)/(true + false positive)
        # Precision = (TP_Count)/(TP_Count+FP_Count)
        # print("Overall Precision: " + str(Precision))
        # opened_file.write(result_value)
        # opened_file.write("\n")
        #
        # # Recall = (true positive)/(true positive + false negative)
        # Recall = (TP_Count)/(TP_Count+FN_Count)
        # print("Overall Recall: " + str(Recall))
        # opened_file.write(result_value)
        # opened_file.write("\n")
        #
        # # F1 = 2*Precision*Recall/(Precision + Recall)
        # F1_score = (2*Precision*Recall) / (Precision + Recall)
        # print("Overall F1 score: " + str(F1_score))
        # opened_file.write(result_value)
        # opened_file.write("\n")
        fpr, tpr, thresholds = skm.roc_curve(y_test, y_score, pos_label=1)
        return fpr, tpr, number_of_trees, subsample_size


# file_name_input = "annthyroid"
# file_name_input = "cardio"
# file_name_input = "covtype"
# file_name_input = "ionosphere"
# file_name_input = "mnist"
# file_name_input = "satellite"
# file_name_input = "shuttle"
# file_name_input = "thyroid"
file_name_input = ["annthyroid","cardio","ionosphere","satellite","shuttle","thyroid"]
# file_name_input = ["ionosphere","satellite","shuttle","thyroid"]


for filename in file_name_input:
    fpr_org, tpr_org, number_of_trees_org, subsample_size_org  = testEIF_Scoring_data(filename, Copula=False)
    fpr_copula, tpr_copula, number_of_trees, subsample_size = testEIF_Scoring_data(filename, Copula=True)
    fig, ax = plt.subplots()
    ax.plot(fpr_org, tpr_org, label="origin")
    ax.plot(fpr_copula, tpr_copula, label="copula")
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title("ROC Curve")
    ax.legend()
    plot_path = "./plots/" + filename + "_ROC_Curve" + str(number_of_trees) + "-" + str(subsample_size) +".jpg"
    fig.savefig(plot_path)
    plt.close(fig)
