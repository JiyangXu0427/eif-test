import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


def logistic_regress_on_original_data(folder_name, file_name):
    raw_data = pd.read_excel("./" + folder_name + "/" + file_name + ".xlsx")
    raw_data_anomaly = raw_data[raw_data["label"] == 1]
    raw_data_normal = raw_data[raw_data["label"] == 0]
    number_of_anomaly_sample = raw_data_anomaly.shape[0]
    number_of_normal_sample = raw_data_normal.shape[0]
    print("Original number of anomaly: " + str(number_of_anomaly_sample))
    print("Original number of normal: " + str(number_of_normal_sample))

    anomaly_min_ratio = 0.25
    if number_of_normal_sample / number_of_anomaly_sample > (1 / anomaly_min_ratio):
        sample_ratio = round(
            (number_of_anomaly_sample * (1 - anomaly_min_ratio) / anomaly_min_ratio) / number_of_normal_sample, 2)
        raw_data_normal = raw_data_normal.sample(frac=sample_ratio)
    print("Undersample number of anomaly: " + str(raw_data_anomaly.shape[0]))
    print("Undersample number of normal: " + str(raw_data_normal.shape[0]))

    with open("./" + folder_name + "/regression_result_" + file_name + "_origin.txt", 'a') as opened_file:
        opened_file.write("Undersample number of anomaly: " + str(raw_data_anomaly.shape[0]))
        opened_file.write("\n")
        opened_file.write("Undersample number of normal: " + str(raw_data_normal.shape[0]))

        raw_data_undersample = pd.concat([raw_data_anomaly, raw_data_normal], axis=0)

        # shuffle
        raw_data_undersample = raw_data_undersample.sample(frac=1).reset_index(drop=True)

        # 筛选特征值和目标值
        x_undersample = raw_data_undersample.loc[:, ["mean", "Standard_Deviation", "Skewness"]]
        y_undersample = raw_data_undersample["label"]

        x_train, x_test, y_train, y_test = train_test_split(x_undersample, y_undersample)

        # 4、标准化
        from sklearn.preprocessing import StandardScaler

        transfer = StandardScaler()
        x_train = transfer.fit_transform(x_train)
        x_test = transfer.transform(x_test)

        from sklearn.linear_model import LogisticRegression

        # 5、预估器流程
        estimator = LogisticRegression()
        estimator.fit(x_train, y_train)
        y_predict = estimator.predict(x_test)

        # 6、模型评估
        # 计算准确率
        score = estimator.score(x_test, y_test)
        # print(score)
        # print(estimator.coef_)
        # print(estimator.intercept_)

        # 查看精确率、召回率、F1-score
        from sklearn.metrics import classification_report

        report = classification_report(y_test, y_predict, labels=[0, 1], target_names=["Normal", "Anormaly"])
        print(report)
        opened_file.write("\n")
        opened_file.write("")
        opened_file.write(report)

        import sklearn.metrics as skm

        skm.plot_roc_curve(estimator, x_test, y_test)
        plt.show()

        roc_auc_score_result = skm.roc_auc_score(y_test, y_predict)
        print("ROC_AUC_Score: " + str(roc_auc_score_result))
        opened_file.write("\n")
        opened_file.write("")
        opened_file.write("ROC_AUC_Score: " + str(roc_auc_score_result))

        transfer_for_statsmodels = StandardScaler()
        x_for_statsmodels = transfer_for_statsmodels.fit_transform(x_undersample)

        import statsmodels.api as sm
        log_reg = sm.Logit(y_undersample, x_for_statsmodels).fit()
        print(log_reg.summary())
        opened_file.write("\n")
        opened_file.write("")
        opened_file.write(str(log_reg.summary()))


def logistic_regress_on_normalised_data(folder_name, file_name):
    raw_data = pd.read_excel("./" + folder_name + "/" + file_name + ".xlsx")
    raw_data_anomaly = raw_data[raw_data["label"] == 1]
    raw_data_normal = raw_data[raw_data["label"] == 0]
    number_of_anomaly_sample = raw_data_anomaly.shape[0]
    number_of_normal_sample = raw_data_normal.shape[0]
    print("Original number of anomaly: " + str(number_of_anomaly_sample))
    print("Original number of normal: " + str(number_of_normal_sample))

    anomaly_min_ratio = 0.25
    if number_of_normal_sample / number_of_anomaly_sample > (1 / anomaly_min_ratio):
        sample_ratio = round(
            (number_of_anomaly_sample * (1 - anomaly_min_ratio) / anomaly_min_ratio) / number_of_normal_sample, 2)
        raw_data_normal = raw_data_normal.sample(frac=sample_ratio)
    print("Undersample number of anomaly: " + str(raw_data_anomaly.shape[0]))
    print("Undersample number of normal: " + str(raw_data_normal.shape[0]))

    with open("./" + folder_name + "/regression_result_" + file_name + "_normalised.txt", 'a') as opened_file:
        opened_file.write("Undersample number of anomaly: " + str(raw_data_anomaly.shape[0]))
        opened_file.write("\n")
        opened_file.write("Undersample number of normal: " + str(raw_data_normal.shape[0]))

        raw_data_undersample = pd.concat([raw_data_anomaly, raw_data_normal], axis=0)

        # shuffle
        raw_data_undersample = raw_data_undersample.sample(frac=1).reset_index(drop=True)

        # 筛选特征值和目标值
        x_undersample = raw_data_undersample.loc[:, ["mean_normalised", "Standard_Deviation_normalised", "Skewness"]]
        y_undersample = raw_data_undersample["label"]

        x_train, x_test, y_train, y_test = train_test_split(x_undersample, y_undersample)

        # 4、标准化
        from sklearn.preprocessing import StandardScaler

        transfer = StandardScaler()
        x_train = transfer.fit_transform(x_train)
        x_test = transfer.transform(x_test)

        from sklearn.linear_model import LogisticRegression

        # 5、预估器流程
        estimator = LogisticRegression()
        estimator.fit(x_train, y_train)
        y_predict = estimator.predict(x_test)

        # 6、模型评估
        # 计算准确率
        score = estimator.score(x_test, y_test)
        # print(score)
        # print(estimator.coef_)
        # print(estimator.intercept_)

        # 查看精确率、召回率、F1-score
        from sklearn.metrics import classification_report

        report = classification_report(y_test, y_predict, labels=[0, 1], target_names=["Normal", "Anormaly"])
        print(report)
        opened_file.write("\n")
        opened_file.write("")
        opened_file.write(report)

        import sklearn.metrics as skm

        skm.plot_roc_curve(estimator, x_test, y_test)
        plt.show()

        roc_auc_score_result = skm.roc_auc_score(y_test, y_predict)
        print("ROC_AUC_Score: " + str(roc_auc_score_result))
        opened_file.write("\n")
        opened_file.write("")
        opened_file.write("ROC_AUC_Score: " + str(roc_auc_score_result))

        transfer_for_statsmodels = StandardScaler()
        x_for_statsmodels = transfer_for_statsmodels.fit_transform(x_undersample)

        import statsmodels.api as sm
        log_reg = sm.Logit(y_undersample, x_for_statsmodels).fit()
        print(log_reg.summary())
        opened_file.write("\n")
        opened_file.write("")
        opened_file.write(str(log_reg.summary()))


file_name = "cardio_statistic_data"
folder_name = "logistic_regression_data_normalised"
logistic_regress_on_normalised_data(folder_name,file_name)