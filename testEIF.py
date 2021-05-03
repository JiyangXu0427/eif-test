import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import multivariate_normal
import random as rn
import eif as iso
import seaborn as sb
sb.set_style(style="whitegrid")
sb.set_color_codes()
import scipy.ndimage
from scipy.interpolate import griddata
import numpy.ma as ma
from numpy.random import uniform, seed
import eif_old as eif_old_class
from sklearn.ensemble import IsolationForest
import pandas as pd

# Generate Artificial data
mean = [0, 0]
cov = [[1, 0], [0, 1]]  # diagonal covariance
Nobjs = 500
np.random.seed(1)
x, y = np.random.multivariate_normal(mean, cov, Nobjs).T

#Add manual outlier
x[0]=3.3
y[0]=3.3
raw_datas=np.array([x,y]).T


#train the forest
number_of_trees = 200
subsample_size = 256
F1  = eif_old_class.iForest(raw_datas, ntrees=number_of_trees, sample_size=subsample_size, ExtensionLevel=1)

#compute score for testing data
testing_data = raw_datas # here we use traning data to see their score, can be replaced by test set
S, LengthDistributions = F1.compute_paths(X_in=testing_data)
print("computing done")

#split the result into anomaly and normal lists
#the threshold can be modified from 0 - 1
threshold = 0.5
Anomalys = []
Normal_Data = []
for i in range(len(LengthDistributions)):
    if LengthDistributions[i][-1] > threshold:
        Anomalys.append(LengthDistributions[i])
    else:
        Normal_Data.append(LengthDistributions[i])

#plot the histogram for the path length of anomaly and normal data
if Anomalys != None:
    print("Number of Anomaly:" + str(len(Anomalys)))
    for j in range(len(Anomalys)):
        sb.histplot(Anomalys[j][1:-1], kde=True, color="b")
        plt.title('extended')
        score = round(Anomalys[j][-1],2)
        path = "./plots/" + "A" + "-" + str(Anomalys[j][0]) + "-" + str(score) + ".jpg"
        plt.savefig(path)
        plt.close()
        print("Anomaly " + str(j) + " plot done")

if Normal_Data != None:
    print("Number of normal data:" + str(len(Normal_Data)))
    for j in range(len(Normal_Data)):
        sb.histplot(Normal_Data[j][1:-1], kde=True, color="b")
        plt.title('extended')
        path = "./plots/" + "N" + "-" + str(Normal_Data[j][0]) + "-" + str(round(Normal_Data[j][-1],2)) + ".jpg"
        plt.savefig(path)
        plt.close()
        print("Normal " + str(j) + " plot done")

#draw a map to show 10 highest score points in black and 10 lowest score points in red
ss1=np.argsort(S)
plt.subplot(1,2,2)
plt.scatter(x,y,s=15,c='b',edgecolor='b')
plt.scatter(x[ss1[-10:]],y[ss1[-10:]],s=55,c='k')
plt.scatter(x[ss1[:10]],y[ss1[:10]],s=55,c='r')
plt.title('extended')
plt.show()


###################Standard Isolation Forest###############################
# rng = np.random.RandomState(42)
#
# # Generate train data
# X = 0.3 * rng.randn(100, 2)
# X_train = np.r_[X + 2, X - 2]
# # Generate some regular novel observations
# X = 0.3 * rng.randn(20, 2)
# X_test = np.r_[X + 2, X - 2]
# # Generate some abnormal novel observations
# X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))
#
# # fit the model
# clf = IsolationForest(max_samples=100, random_state=rng)
# clf.fit(X_train)
# y_pred_train = clf.predict(X_train)
# y_pred_test = clf.predict(X_test)
# y_pred_outliers = clf.predict(X_outliers)
#
# print(y_pred_train)
# print(y_pred_test)
# print(y_pred_outliers)