import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from MAEGAN.model.MAEGAN import MAEGAN
import matplotlib.pyplot as plt

def find_anomaly_ip(anomaly_flow_with_preds, save_path):
    df = pd.DataFrame(
        anomaly_flow_with_preds.reshape(-1, 5),  # 展平到 2D 形状，每个样本有 5 个属性
        columns=["ipv4_initiator", "ipv4_responder", "start_time", "0_pred", "1_pred"]
    )
    df["1_pred"] = pd.to_numeric(df["1_pred"], errors="coerce")

    top_ips = (
        df.groupby("ipv4_initiator")["1_pred"]
        .mean()
        .nlargest(10)  # 获取 1_pred 累计值最大的前 top_n 个 IP
        .reset_index()
    )
    print(top_ips)
    top_ips.to_csv(save_path, index=False)
    print(f"Top IPs saved to {save_path}")
    
class OCSVM_ID():
    def __init__(self):
        self.model = OneClassSVM()
    
    def train(self, data_embs):
        self.model.fit(data_embs)
    
    def predict(self, args, data_embs, flow_with_preds, percentage = 0.05):
        scores = self.model.decision_function(data_embs)#值越低越不正常
        anomaly_scores = -scores
        num_samples = int(len(anomaly_scores) * percentage)
        anomaly_indices = np.argsort(scores)[-num_samples:] #选择异常概率最大的几个样本
        anomaly_flow_with_preds = flow_with_preds[anomaly_indices]

        save_path = os.path.join('DyGCN/data/', args.dataset, 'top_ips.csv')
        find_anomaly_ip(anomaly_flow_with_preds, save_path)

class Options:
    def __init__(self):
        self.n_epochs = 80
        self.batch_size = 64
        self.lr = 0.0001
        self.b1 = 0.5
        self.b2 = 0.9
        self.n_critic = 5
        self.seed = 42

class MAEGAN_Detect():
    def __init__(self, config):
        self.device = config["device"]

        config = config["maegan"]
        opt = Options()
        self.filepath = filepath
        self.datapath = datapath
        self.model = MAEGAN(opt, input_dim = input_dim, filepath=filepath, datapath=datapath)

    def train(self, data_embs):
        self.model.train(data_embs)
        self.model.save()

    def predict(self, args, data_embs, flow_with_preds, result_path, percentage = 0.05):
        self.model = MAEGAN.load(self.filepath)
        scores = self.model.test(data_embs, np.zeros(data_embs.shape[0]))
        num_samples = int(len(scores) * percentage)
        anomaly_indices = np.argsort(scores)[-num_samples:] #选择异常概率最大的几个样本
        anomaly_flow_with_preds = flow_with_preds[anomaly_indices]

        find_anomaly_ip(anomaly_flow_with_preds, os.path.join(self.datapath, 'top_ips.csv'))
