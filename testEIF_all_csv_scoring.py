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


def testEIF_Scoring_data(filename, number_of_trees, subsample_size, extensionLevel , Copula=False):

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

    if subsample_size == "half":
        subsample_size = int(len(raw_datas_without_label) / 2)

    if subsample_size > int(len(raw_datas_without_label) / 2) :
        subsample_size = int(len(raw_datas_without_label) / 2)



    if extensionLevel == "full":
        extd_level_in_filename = extensionLevel
        extensionLevel = raw_datas_without_label.shape[1] - 1
    else:
        extd_level_in_filename = str(extensionLevel)

    # traing the forest
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
        result_data_path = "./plots/" + filename + "_Classification_Result_Data_Org-"+ str(number_of_trees) + "-" + str(subsample_size) + "-" + extd_level_in_filename + ".xlsx"
    else:
        result_data_path = "./plots/" + filename + "_Classification_Result_Data_Copula-"+ str(number_of_trees) + "-" + str(subsample_size) + "-" + extd_level_in_filename +".xlsx"

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
            opened_file.write("Extend Level: " + extd_level_in_filename)
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

        fpr, tpr, thresholds = skm.roc_curve(y_test, y_score, pos_label=1)
        return fpr, tpr, subsample_size


# filename = "annthyroid"
# filename = "cardio"
# filename = "covtype"
# filename = "ionosphere"
# filename = "mnist"
# filename = "satellite"
# filename = "shuttle"
# filename = "thyroid"
# filenames = ["annthyroid","cardio","ionosphere","satellite","shuttle","thyroid"]
# filenames = ["ionosphere","satellite","shuttle","thyroid"]
# filenames = ["annthyroid","cardio","satellite","shuttle","thyroid"]
# filenames = ["cardio","thyroid"]
# filenames = ["shuttle"]
# filenames = ["satellite"]
filenames = ["ionosphere"]
# parameter for traing the forest
number_of_trees = 100
subsample_size = 256
extensionLevel = 1

for filename in filenames:
    fpr_org, tpr_org, subsample_size_org  = testEIF_Scoring_data(filename, number_of_trees, subsample_size, extensionLevel, Copula=False)
    fpr_copula, tpr_copula, subsample_size_copula = testEIF_Scoring_data(filename, number_of_trees, subsample_size, extensionLevel, Copula=True)
    fig, ax = plt.subplots()
    ax.plot(fpr_org, tpr_org, label="origin")
    ax.plot(fpr_copula, tpr_copula, label="copula")
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title("ROC Curve")
    ax.legend()
    if extensionLevel != "full":
        extensionLevel = str(extensionLevel)
    plot_path = "./plots/" + filename + "_ROC_Curve" + str(number_of_trees) + "-" + str(subsample_size_copula) + "-" + extensionLevel +".jpg"
    fig.savefig(plot_path)
    plt.close(fig)
    extensionLevel = int(extensionLevel)
