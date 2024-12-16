import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score, accuracy_score, precision_score, recall_score, classification_report

import matplotlib.pyplot as plt
cur_path=os.path.dirname(__file__)
os.chdir(cur_path)


def visual(labels, anomaly_score, edges):
    plt.clf()
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC-AUC")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    ########################################
    # # 计算 ROC 曲线
    # fpr, tpr, thresholds = roc_curve(labels, anomaly_score)
    # # 找到最优阈值
    # optimal_idx = np.argmax(tpr - fpr)
    # optimal_threshold = thresholds[optimal_idx]
    # print(optimal_threshold)

    # # 使用最优阈值来生成预测标签
    # predicted_labels = np.where(anomaly_score >= optimal_threshold, 1, 0)
    # print("roc_curve")
    # print(classification_report(labels, predicted_labels))


    precision, recall, thresholds = precision_recall_curve(labels, anomaly_score)
    f1_scores = np.where((precision + recall) == 0, 0, 2 * (precision * recall) / (precision + recall))
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    print(optimal_threshold)

    # 根据最优阈值生成预测标签
    predicted_labels = np.where(anomaly_score >= optimal_threshold, 1, 0)

    # 找到所有被预测为异常的样本索引
    anomaly_indices = np.where(predicted_labels == 1)[0]

    # 提取这些样本的实际标签
    actual_edges = np.array(edges)[anomaly_indices]

    # 输出每个被预测为异常的样本的实际标签
    print("Predicted Anomalies and Their Actual Labels:")
    for idx, actual_label in zip(anomaly_indices, actual_edges):
        print(f"Sample {idx}: Actual Label = {actual_label}")


def parse():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cic2018', type=str)
    return parser.parse_args()


def plot_roc(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    maxindex = (tpr-fpr).tolist().index(max(tpr-fpr))
    threshold = thresholds[maxindex]
    print('异常阈值', threshold)
    auc_score = auc(fpr, tpr)
    print('auc值: {:.4f}'.format(auc_score))
    return threshold, auc_score


def eval(labels,pred):
    plot_roc(labels, pred)
    print(confusion_matrix(labels, pred))
    a,b,c,d=accuracy_score(labels, pred),precision_score(labels, pred),recall_score(labels, pred), f1_score(labels, pred)
    print("acc:{:.4f},pre{:.4f},rec:{:.4f}, f1:{:.4f}".format(a,b,c,d))
    return a,b,c,d


def matrix(scores, records, min_score=None, max_score=None):
    print(min_score, max_score)

    # 进行 Min-Max 归一化
    scores = (scores - min_score) / (max_score - min_score)
    print(scores)

    indices_above_1 = np.where(scores > 1)[0]
    records_above_1 = records[indices_above_1]

    records_df = pd.DataFrame(
        records_above_1.reshape(-1, 5),  # 展平到 2D 形状，每个样本有 5 个属性
        columns=["ipv4_initiator", "ipv4_responder", "start_time", "0_pred", "1_pred"]
    )
    
    # 按 ipv4_initiator 分组并累计 1_pred
    records_df["1_pred"] = pd.to_numeric(records_df["1_pred"], errors="coerce")

    top_ips = (
        records_df.groupby("ipv4_initiator")["1_pred"]
        .mean()
        .nsmallest(10)  # 获取 1_pred 累计值最大的前 top_n 个 IP
        .reset_index()
    )

    print("累计 1_pred 最大的前几个 IP:")
    print(top_ips)

if __name__ =='__main__':
    args=parse()
    seq_len=5
    dataset=args.dataset
    embs_path = os.path.join('data', "graph_embs.pt")
    # embs_path='data/graph_embs.pt'
    labels_path = os.path.join('data', "labels.npy")
    edges_path = os.path.join('data', "edges.npy")
    # if dataset=='data/cic2017':
    #     train_len=[0, 1500]
    # else:
    #     train_len=[0, 4600] #cic[0,529], unsw[200:600], ustc[10:100]
    train_len=[0, 2000]
    args.in_dim=69

    data_embs = torch.load(embs_path).detach().cpu().numpy()
    print(len(data_embs))
    print(data_embs.shape)
    print(data_embs[train_len[0]+seq_len:train_len[1]].shape)
    print(np.concatenate([data_embs[:train_len[0]],data_embs[train_len[1]:]]).shape)
    labels = np.load(labels_path, allow_pickle=True)
    labels=labels[seq_len:]
    labels=np.concatenate((labels[:train_len[0]], labels[train_len[1]:]))

    records = pd.read_csv("/root/GCN/DyGCN/data/flow_with_preds.csv")
    records = records.values.reshape(-1,1000,5)
    records=records[seq_len:]
    records=np.concatenate((records[:train_len[0]], records[train_len[1]:]))
    iof = OneClassSVM()
    iof=iof.fit(data_embs[train_len[0]+seq_len:train_len[1]])



    # 将 test_embs 拆分成前半部分和后半部分
    test_embs = np.concatenate([data_embs[:train_len[0]], data_embs[train_len[1]:]])
    split_index = 3000   # 使用前半部分计算 min_score 和 max_score
    print(len(test_embs))
    score_calibration_part = test_embs[:split_index]
    score_evaluation_part = test_embs[split_index:]

    # 计算 min_score 和 max_score
    calibration_scores = iof.decision_function(score_calibration_part)
    min_score = np.min(calibration_scores)
    max_score = np.max(calibration_scores)

    # 计算 evaluation 部分的 scores 并应用 matrix 函数
    evaluation_scores = iof.decision_function(score_evaluation_part)
    matrix(-evaluation_scores, records[split_index:], min_score, max_score)
    
    
    # np.save(dataset+'scores.npy', -scores)    
    # pred = torch.zeros(len(scores))
    # idx=scores.argsort()#从大到小


    # vs=[aucv*100, pre*100, rec*100, f1*100]
    # for k in range(500,2400,500):
    #     print('============ k=',k)
    #     pred[idx[:k]]=1
    #     a,b,c,d=eval(labels.astype(np.long), pred)
    #     # print("acc:{:.4f},pre{:.4f},rec:{:.4f}, f1:{:.4f}".format(a,b,c,d))
    #     vs+=[b*100,c*100]
    #     # df.append([b,c])
    # print(vs)
    # df = pd.DataFrame([vs])
    # df.to_csv('rgcn-o-osvm.csv')
