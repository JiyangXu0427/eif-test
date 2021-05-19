import matplotlib.pyplot as plt
import seaborn as sb
sb.set_style(style="whitegrid")
sb.set_color_codes()
import numpy as np
from scipy.stats import skew
import pandas as pd
import eif_old as eif_old_class

#read data
raw_datas = np.loadtxt("./datasets/covtype.data.txt",delimiter=",",unpack=False)
raw_datas_without_label = raw_datas[:,:-1]

#train the forest
number_of_trees = 400
subsample_size = int(len(raw_datas)/2)
F1  = eif_old_class.iForest(raw_datas_without_label, ntrees=number_of_trees, sample_size=subsample_size, ExtensionLevel=1)

#compute score for testing data
testing_data = raw_datas_without_label # here we use traning data to see their score, can be replaced by test set
S, DataPointsResult = F1.compute_paths(X_in=testing_data)
print("computing done")

#split the result into anomaly and normal lists
#the threshold can be modified from 0 - 1
threshold = 0.5
Anomalys = []
Normal_Data = []
for i in range(len(DataPointsResult)):
    if DataPointsResult[i]["score"] > 0.5:
        Anomalys.append(DataPointsResult[i])
    else:
        Normal_Data.append(DataPointsResult[i])

#plot the histogram for the path length of anomaly and normal data
a_means = []
a_variances =[]
a_std_devs = []
a_skws = []

if Anomalys != None:
    print("Number of Anomaly:" + str(len(Anomalys)))
    for j in range(len(Anomalys)):
        lengths = Anomalys[j]["length"]
        a_means.append(np.mean(lengths))
        a_variances.append(np.var(lengths))
        a_std_devs.append(np.std(lengths))
        a_skws.append(skew(lengths))

        sb.histplot(lengths, kde=True, color="b")
        plt.title('Anomaly')
        score = round(Anomalys[j]["score"],2)
        path = "./plots/" + str(score) + "-" + "A" + "-" + str(Anomalys[j]["list_num"]) + ".jpg"
        plt.savefig(path)
        plt.close()
        print("Anomaly " + str(j) + " plot done")

n_means = []
n_variances =[]
n_std_devs = []
n_skws = []

if Normal_Data != None:
    print("Number of normal data:" + str(len(Normal_Data)))
    for j in range(len(Normal_Data)):
        lengths = Normal_Data[j]["length"]
        n_means.append(np.mean(lengths))
        n_variances.append(np.var(lengths))
        n_std_devs.append(np.std(lengths))
        n_skws.append(skew(lengths))

        sb.histplot(lengths, kde=True, color="b")
        plt.title('Normal')
        score = round(Normal_Data[j]["score"],2)
        path = "./plots/" + str(score) + "-" + "N" + "-" + str(Normal_Data[j]["list_num"]) + ".jpg"
        plt.savefig(path)
        plt.close()
        print("Normal " + str(j) + " plot done")

anomalyDataFrame = pd.DataFrame({"mean":a_means,"variance": a_variances,"Standard_Deviation":a_std_devs,"Skewness":a_skws})

nomalDataFrame = pd.DataFrame({"mean":n_means,"variance": n_variances,"Standard_Deviation":n_std_devs,"Skewness":n_skws})

anomalyDataFrame.to_excel("./plots/anomaly_statistic.xlsx")
nomalDataFrame.to_excel("./plots/normal_statistic.xlsx")