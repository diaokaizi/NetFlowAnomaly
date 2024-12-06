import torch
from torchvision import datasets
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score, accuracy_score, precision_score, recall_score, classification_report
import matplotlib.pyplot as plt
import os
class NormalizeTransform:
    """ Normalize features with mean and standard deviation. """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        return (sample - self.mean) / self.std

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform=None):
        self.transform = transform
        self.data = data
        self.labels = labels

    def __getitem__(self, idx):
        data = self.data[idx]
        labels = self.labels[idx]

        if self.transform:
            data = self.transform(data)

        return data, labels

    def __len__(self):
        return len(self.data)

def fix_name():
    return ["sportsocks", "sportprivate", "dportirc", "sporttelnet", "sportrapservice", "dporthttp",
            "sportsyslog", "sportreserved", "dportkpasswd", "tcpflagsACK", "npacketsmedium",
            "sportcups", "dporttelnet", "sportldaps", "tcpflagsPSH", "dportoracle"]

def load_UGR16():
    selected_feature_names = fix_name()
    raw_x_train = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Xtrain.csv").drop(columns=["Row"], axis=1)
    x_train = raw_x_train[selected_feature_names]
    x_train = torch.from_numpy(x_train.values).float()
    y_train = torch.zeros(len(x_train))


    raw_x_test = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Xtest.csv").drop(columns=["Row"], axis=1)
    x_test = raw_x_test[selected_feature_names]
    x_test = torch.from_numpy(x_test.values).float()
    y_test = pd.read_csv("/root/bishe/dataset/URD16/UGR16v1.Ytest.csv").drop(columns=["Row", "labelanomalyidpscan", "labelanomalysshscan", "labelanomalyidpscan", "labelblacklist"], axis=1)
    y_test = torch.from_numpy(y_test.apply(lambda row: 1 if row.sum() > 0 else 0, axis=1).values)
    return (x_train, y_train), (x_test, y_test)

def load_UGR16_gas():
    raw_x_train = pd.read_csv("/root/GSA-AnoGAN/KitNet/gsa.csv", header=None).drop(columns=[0], axis=1)
    x_train = torch.from_numpy(raw_x_train.values).float()
    y_train = torch.zeros(len(x_train))
    return (x_train, y_train), (0, 0)

def report_result(name, labels, anomaly_score, fun="pr"):
    print("XXXXXXXXXXXXX")
    print(name)
    if fun == "roc":
        fpr, tpr, thresholds = roc_curve(labels, anomaly_score)
        # 找到最优阈值
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
    else:
        precision, recall, thresholds = precision_recall_curve(labels, anomaly_score)
        f1_scores = np.where((precision + recall) == 0, 0, 2 * (precision * recall) / (precision + recall))
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
    # 根据最优阈值生成预测标签
    predicted_labels = np.where(anomaly_score >= optimal_threshold, 1, 0)
    print("precision_recall_curve")
    # print(classification_report(labels, predicted_labels))
    report_dict = classification_report(labels, predicted_labels, output_dict=True)

    # 手动格式化输出，保留4位小数
    for label, metrics in report_dict.items():
        if label == "1":
            if isinstance(metrics, dict):  # 这是每个标签的指标部分
                print(f"Label: {label}")
                for metric, score in metrics.items():
                    print(f"  {metric}: {score:.4f}")
            else:  # 这是整体的指标部分，如 'accuracy'
                print(f"{label}: {metrics:.4f}")
    fpr, tpr, _ = roc_curve(labels, anomaly_score)
    precision, recall, _ = precision_recall_curve(labels, anomaly_score)
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    plt.plot(fpr, tpr, label=f"{name} = {roc_auc:3f}")
    plt.title("ROC-AUC")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(os.path.join(name, "ROC-AUC.png"))
