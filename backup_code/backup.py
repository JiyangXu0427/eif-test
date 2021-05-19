# for i in range(len(DataPointsResult)):
#     if DataPointsResult[i]["score"] > 0.5:
#         True_Anomalys.append(DataPointsResult[i])
#     else:
#         True_Normal_Data.append(DataPointsResult[i])

# plot the histogram for the path length of anomaly and normal data

# num_bins = 50
# fig, ax = plt.subplots()
# n, bins, patches = ax.hist(lengths, num_bins, density=True)
# plt.close(fig)

# add a 'best fit' line
# y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
#      np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2))
# axTotal.plot(bins, y, linestyle='--', label=str(score) + "-" + "A" + "-" + str(Anomalys_by_label[j]["list_num"]))

# sb.histplot(lengths, kde=True, color="b")
# plt.title('Anomaly')
# score = round(True_Anomalys[j]["score"], 2)
# path = "./plots/" + str(score) + "-" + "A" + "-" + str(True_Anomalys[j]["list_num"]) + ".jpg"
# plt.savefig(path)
# plt.close()
# print("Anomaly " + str(j) + " plot done")



# y_pre = [1,	2,	1,	1,	3,	4,	3,	2,	3,	2,	2,	1,	2,	4,	4,	2,	3,	2,	2,	2,	2,	1,	1,	4,	4,	2,	1,	3,	2,	1]
#
# y_true= [1,	1,	3,	3,	3,	4,	3,	2,	2,	3,	3,	1,	4,	3,	4,	3,	3,	2,	4,	2,	1,	1,	1,	4,	1,	3,	1,	1,	1,	1]

y_pre = [1,	0,	1,	1,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	1,	0,	0,	0,	1,	0,	0,	1]
y_true= [1,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	1,	1,	1,	0,	1,	0,	1,	1,	1,	1]


result_value = skm.accuracy_score(y_true, y_pre)
print("Accuracy_Score: " + str(result_value))

result_value = skm.precision_score(y_true, y_pre)
print("Precision Score: " + str(result_value))

result_value = skm.recall_score(y_true, y_pre)
print("Recall Score: " + str(result_value))

result_value = skm.f1_score(y_true, y_pre)
print("F1 Score: " + str(result_value))

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