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