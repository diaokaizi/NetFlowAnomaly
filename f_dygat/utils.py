import argparse
import math
import random
from operator import neg

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_curve)
from torch.nn.functional import softmax
from torch.utils.data import dataset
import tqdm

class FlowDataset(dataset.Dataset):
    def __init__(self, data):
        self.data=data
    def __getitem__(self, index):
        return self.data[index], self.data[index]
    def __len__(self):
        return len(self.data)

class CrossEntropyLoss(nn.Module):
    def forward(self, x, idx_edge):
        '''
        ip_x1:节点的出边嵌入特征
        ip_x2:节点的入边嵌入特征
        pos_rows:正采样的源节点
        pos_cols:正采样的目的节点
        neg_rows:负采样的源节点
        neg_cols:负采样的目的节点
        '''
        pos_rows, pos_cols, neg_rows, neg_cols=idx_edge
        #边的特征:[源节点的入边特征、目的节点的出边特征]  对比损失，有边相连的节点表示尽可能相似
        pos_edges = torch.cat([x[pos_rows], x[pos_cols]], dim=1)
        neg_edges = torch.cat([x[neg_rows], x[neg_cols]], dim=1)
        feats = torch.cat([pos_edges, neg_edges])
        label = torch.cat([torch.ones_like(pos_edges), torch.zeros_like(neg_edges)]).long()
        loss = F.binary_cross_entropy_with_logits(feats, label.float())
        return loss

def plot_train_loss(losses):
    n=len(losses)
    x=[i for i in range(n)]
    plt.plot(x, losses)
    plt.savefig('train_loss.png')

def plot_roc(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    maxindex = (tpr-fpr).tolist().index(max(tpr-fpr))
    threshold = thresholds[maxindex]
    print('异常阈值', threshold)
    auc_score = auc(fpr, tpr)
    print('auc值: {:.4f}'.format(auc_score))
    return threshold

def matrix(true_graph_labels,scores):
    t=plot_roc(true_graph_labels, scores)
    true_graph_labels = np.array(true_graph_labels)
    scores  = np.array(scores)
    pred=np.ones(len(scores))
    pred[scores<t]=0
    print(confusion_matrix(true_graph_labels, pred))
    print("acc:{:.4f},pre{:.4f},rec:{:.4f}, f1:{:.4f}".format(accuracy_score(true_graph_labels, pred),precision_score(true_graph_labels, pred),recall_score(true_graph_labels, pred), f1_score(true_graph_labels, pred)))

def eval(labels,pred):
    plot_roc(labels, pred)
    print(confusion_matrix(labels, pred))
    a,b,c,d=accuracy_score(labels, pred),precision_score(labels, pred),recall_score(labels, pred), f1_score(labels, pred)
    print("acc:{:.4f},pre{:.4f},rec:{:.4f}, f1:{:.4f}".format(a,b,c,d))
    return a,b,c,d
def NegativeSampler(pos_adj):
    '''根据邻接矩阵pos_adj进行负采样，采样不相连节点形成的边,
    sample_num:负采样的边的数目'''
    adj=pos_adj+sp.eye(pos_adj.shape[0]) # A+I,防止采样到自身到自身的边
    unexists_edges = np.argwhere(adj==0.)
    # n = unexists_edges.shape[0]
    # kk = torch.ones(n)
    # sample_num = pos_adj.data.shape[0]
    # idx = kk.multinomial(sample_num)
    return unexists_edges
def reset_param(t):
    stdv = 2. / math.sqrt(t.size(0))
    t.data.uniform_(-stdv,stdv)
class Namespace(object):
    '''
    helps referencing object in a dictionary as dict.key instead of dict['key']
    '''
    def __init__(self, adict):
        self.__dict__.update(adict)

def create_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config_file', default='parameters_example.yaml', type=argparse.FileType(mode='r'), help="包含参数的配置文件")
    return parser
def parse_args(parser):
    args = parser.parse_args()
    if args.config_file:
        data=yaml.load(args.config_file)
        delattr(args, 'config_file')
        arg_dict=args.__dict__
        for key, value in data.items():
            arg_dict[key]=value
    return args

def neg_sample_graphsage(adj, neg_num):
    # 传统负采样策略，graphsage采用的负采样策略
    weights = torch.sum(adj.to_dense(),1)
    hat_a=adj.to_dense().cpu()
    neg_indices=torch.where(hat_a==0)
    dst = weights.multinomial(neg_num, replacement=True) # 多项式分布采样，第一个参数表示采样的数目。采样的是self.weights的位置，self.weights的每个值表示采样到该元素的权重，
    return dst

def neg_sample_degree_similarity(adj, neg_num, out, pos_src):
    """
    基于节点度和节点相似度的综合负采样策略。

    参数：
    - adj: 稀疏邻接矩阵（torch.sparse_coo_tensor）
    - neg_num: 需要采样的负样本数量
    - out: 节点的嵌入表示（形状：[节点数, 嵌入维度]）
    - pos_src: 正样本边的源节点索引（形状：[边数]）

    返回：
    - neg_dst: 负样本的目标节点索引（形状：[neg_num]）
    """
    # 计算每个节点的度数
    degrees = torch.sparse.sum(adj, dim=1).to_dense()  # 形状：[节点数]
    degrees = degrees + 1e-8  # 防止度数为零

    num_nodes = degrees.size(0)

    # 对节点嵌入进行归一化
    out_norm = F.normalize(out, p=2, dim=1)  # 形状：[节点数, 嵌入维度]

    # 获取正样本源节点的嵌入
    src_embeddings = out_norm[pos_src]  # 形状：[正样本数, 嵌入维度]

    # 计算正样本源节点的平均嵌入
    src_mean_embedding = torch.mean(src_embeddings, dim=0, keepdim=True)  # 形状：[1, 嵌入维度]

    # 计算每个节点与正样本源节点平均嵌入的相似度（余弦相似度）
    similarities = torch.sigmoid(torch.mm(out_norm, src_mean_embedding.t()).squeeze())  # 形状：[节点数]

    # 计算综合采样权重（度数 * 相似度）
    weights = degrees * similarities  # 形状：[节点数]

    # 归一化权重，使其和为1
    weights_sum = weights.sum()
    if weights_sum > 0:
        weights = weights / weights_sum
    else:
        weights = torch.ones(num_nodes, device=adj.device) / num_nodes

    # 根据综合权重进行多项式采样
    neg_dst = torch.multinomial(weights, neg_num, replacement=True)  # 形状：[neg_num]

    return neg_dst

def neg_sample(adj, neg_num):
    # 根据节点的度计算负采样分数，两个节点间度越大，越接近，负采样概率越大
    hat_a = torch.eye(adj.shape[0])+adj.cpu()
    hat_a=adj.to_dense().cpu()
    neg_indices=torch.where(hat_a==0)
    degs = hat_a.sum(0)
    degs_src, degs_dst=degs[neg_indices[0]],degs[neg_indices[1]]
    k=torch.min(degs_src,degs_dst)/torch.max(degs_src,degs_dst)
    neg_sam_scores=(degs_src+degs_dst)*k
    
    neg_idx=neg_sam_scores.sort().indices
    neg_idx=neg_idx[-neg_num:] #选取负样本
    return neg_indices[0][neg_idx], neg_indices[1][neg_idx]

def remove_duplicate_edges(pos_src, pos_dst, neg_src, neg_dst):
    # 创建一个集合，包含真实样本的边（正样本），每个边表示为一个元组
    pos_edges = set(zip(pos_src.tolist(), pos_dst.tolist()))

    # 创建新的列表来存储去重后的负样本边
    new_neg_src = []
    new_neg_dst = []

    # 遍历负样本边并去除与正样本边重复的组合
    for src, dst in zip(neg_src.tolist(), neg_dst.tolist()):
        if (src, dst) not in pos_edges:
            new_neg_src.append(src)
            new_neg_dst.append(dst)

    # 将去重后的负样本边转回torch tensor
    neg_src_filtered = torch.tensor(new_neg_src, dtype=torch.long)
    neg_dst_filtered = torch.tensor(new_neg_dst, dtype=torch.long)

    return neg_src_filtered, neg_dst_filtered

def get_edge_embs_dy(out, adj, cls):
    edge_indices = adj._indices()
    pos_src, pos_dst = edge_indices[0], edge_indices[1]
    neg_src, neg_dst = neg_sample_degree_similarity(adj, len(edge_indices[0]), out, pos_dst), neg_sample_degree_similarity(adj, len(edge_indices[0]), out, pos_src)
    neg_src, neg_dst = remove_duplicate_edges(pos_src, pos_dst, neg_src, neg_dst)

    #边的特征:[源节点的出边特征、目的节点的入边特征]
    pos_edges = torch.cat([out[pos_src], out[pos_dst]], dim=1)
    neg_edges = torch.cat([out[neg_src], out[neg_dst]], dim=1)
    edge_embs = torch.cat([pos_edges, neg_edges])
    label = torch.cat([torch.ones(len(pos_edges)), torch.zeros(len(neg_edges))]).long()
    return edge_embs, label, pos_edges

def get_edge_embs(out, adj, abalation=False):
    edge_indices = adj._indices()
    pos_src, pos_dst = edge_indices[0], edge_indices[1]

    if not abalation:
        neg_src, neg_dst= neg_sample(adj, len(edge_indices[0]))
    else:
        # neg_src, neg_dst = pos_src, neg_sample_graphsage(adj, len(edge_indices[0]))
        # neg_src, neg_dst = neg_sample_graphsage(adj, len(edge_indices[0])), neg_sample_graphsage(adj, len(edge_indices[0]))
        # neg_src, neg_dst = neg_sample_graphsage(adj, len(edge_indices[0])), neg_sample_graphsage(adj, len(edge_indices[0]))
        neg_src, neg_dst = pos_src, neg_sample_degree_similarity(adj, len(edge_indices[0]), out, pos_src)


    #边的特征:[源节点的出边特征、目的节点的入边特征]
    pos_edges = torch.cat([out[pos_src], out[pos_dst]], dim=1)
    neg_edges = torch.cat([out[neg_src], out[neg_dst]], dim=1)
    edge_embs = torch.cat([pos_edges, neg_edges])
    label = torch.cat([torch.ones(len(pos_edges)), torch.zeros(len(neg_edges))]).long()
    return edge_embs, label, pos_edges


class Namespace(object):
    '''
    helps referencing object in a dictionary as dict.key instead of dict['key']
    '''
    def __init__(self, adict):
        self.__dict__.update(adict)

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def edge_num_plot(test):
        adj=test["adj_list"]
        dd=dict()
        for k in tqdm(range(len(adj))):
            A=adj[k].to_dense()
            for i in range(len(A)):
                for j in range(i+1, len(A)):
                    v=A[i][j]+A[j][i] #节点之间的边的计数是双向的
                    v=int(v.item())
                    dd[v]=dd.get(v,0)+1
            break
        dd = {k: v / len(adj) for k, v in dd.items()}
        dd[0]=0
        edge_num=dd.keys()
        pinlv=dd.values()
        plt.bar(edge_num,pinlv)
        plt.show()
        plt.savefig('bar.png')
        print(dd)