import numpy as np
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt

directory = "./data/"


def preprocess_data(data):
    anomalous_data = data[data["anomalous"] == 1]
    sensitivity = len(anomalous_data[anomalous_data["score"] > r]) / len(anomalous_data)

    non_anomalous_data = data[data["anomalous"] == 0]
    specificity = len(non_anomalous_data[non_anomalous_data["score"] < r]) / len(non_anomalous_data)

    data["y"] = data["score"] > r

    return anomalous_data, non_anomalous_data, data


def caclulate_roc_auc(data):
    fpr, tpr, thresholds = metrics.roc_curve(data["anomalous"], data["score"])
    auc = metrics.roc_auc_score(data["anomalous"], data["score"])
    roc = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc)

    return auc, roc


fig, ax = plt.subplots(3, 3, figsize=(10, 10), constrained_layout=True)

for r in range(1, 10):
    data = pd.DataFrame()
    data["input"] = pd.read_csv(directory + "english_tagalog.test", header=None)  # input sentence
    data["score"] = pd.read_csv(directory + "res_" + str(r) + ".txt", header=None).astype(np.float32)  # anomaly score
    data["anomalous"] = 1
    data.loc[:123, "anomalous"] = 0

    _, _, data = preprocess_data(data)

    auc, roc = caclulate_roc_auc(data)
    axis = ax[(r - 1) // 3, (r - 1) % 3]
    roc.plot(ax=axis)
    axis.set_title(f"r={r}")

languages = ["xhosa", "hiligaynon", "middle_english", "plaudietsch"]
fig, ax = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)
i = 0
# java -jar negsel2.jar -self english.train -n 10 -l -c -r 3

for language in languages:
    print(language)
    data = pd.DataFrame()
    data["input"] = pd.read_csv(directory + "english_" + language + ".test", header=None)  # input sentence
    data["score"] = pd.read_csv(directory + language + "_res.txt", header=None).astype(np.float32)  # anomaly score
    data["anomalous"] = 1
    data.loc[:123, "anomalous"] = 0

    _, _, data = preprocess_data(data)

    auc, roc = caclulate_roc_auc(data)
    axis = ax[i // 2, i % 2]
    i += 1
    roc.plot(ax=axis)
    axis.set_title(language)
