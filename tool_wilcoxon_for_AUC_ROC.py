import matplotlib.pyplot as plt
import seaborn as sb

sb.set_style(style="whitegrid")
sb.set_color_codes()
import numpy as np
import scipy.stats as sts
import pandas
import pandas as pd
import eif_old as eif_old_class
import eif as eif_new
import scipy.io as sio
import os
import sklearn.metrics as skm
from sklearn.ensemble import IsolationForest
import math

path = "./EIF_SIF_Result/EIF_SIF_ensemble_ROC_AUC.xlsx"
pd_data = pd.read_excel(path, index_col=0)

EIF_matrix = pd_data[pd_data["algo_name"] == "EIF" ]
EIF_matrix = EIF_matrix[EIF_matrix["dataset_name"] != "speech" ]

SIF_matrix = pd_data[pd_data["algo_name"] == "SIF" ]
SIF_matrix = SIF_matrix[SIF_matrix["dataset_name"] != "speech" ]

np_EIF_AUC_Vector = np.array(EIF_matrix["ROC_AUC"])
np_SIF_AUC_Vector = np.array(SIF_matrix["ROC_AUC"])

w_all,p_all = sts.wilcoxon(np_EIF_AUC_Vector,np_SIF_AUC_Vector)
print(w_all)
print(p_all)






#
# np_dataset_name = np.array(SIF_AUC_Vector["dataset_name"])
# np_dataset_type = np.array(SIF_AUC_Vector["data_type"])
# wilcoxon_result = pd.DataFrame({"dataset_name":np_dataset_name,"data_type":np_dataset_type,"EIF_VS_SIF_statistic":w,"EIF_VS_SIF_pvalue":p})

# print(np_EIF_AUC_Vector)
# print(np_SIF_AUC_Vector)
# print(wilcoxon_result)