import datetime
import glob
import math
import os
from collections import Counter, defaultdict
from time import time
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt

class GraphData():
    def __init__(self, Ain_list, Aout_list, A_list, adj_list, feat_list, ip_list, flow_list):
        self.Ain_list=Ain_list
        self.Aout_list = Aout_list
        self.A_list = A_list
        self.adj_list = adj_list
        self.feat_list = feat_list
        self.ip_list = ip_list
        self.flow_list = flow_list

class Dataset():
    def __init__(self, df_list, flow_num):
        self.adj_list=[]
        self.label_list=[]
        self.Ain_list=[]
        self.Aout_list=[]
        self.A_list=[]
        self.ip_list=[] # 每个图的ip
        self.node_feats={} # 随机生成的节点特征
        self.df_list = df_list
        self.flow_num = flow_num
        self.df = None
        self.gen_graphs()
    
    def normalize_random(self,adj):
        """邻接矩阵归一化：随机游走归一化"""
        # adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        rowsum[rowsum == 0] = 1
        d_inv_sqrt = np.power(rowsum, -1).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(torch.from_numpy(d_inv_sqrt))
        adj=torch.mm(d_mat_inv_sqrt, adj)
        return adj
    
    def normalize_sym(self,adj):
        """邻接矩阵归一化：对称归一化"""
        # adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(torch.from_numpy(d_inv_sqrt))
        adj=torch.mm(d_mat_inv_sqrt, adj)
        return torch.mm(adj, d_mat_inv_sqrt)
    
    def gen_node_feats(self, ips):
        for ip in ips:
            if ip not in self.node_feats:
                self.node_feats[ip]=torch.randn(77)

    def gen_graphs(self):
        self.process_cst()

        le = LabelEncoder()
        edges = self.flow_list.values.reshape(-1, self.flow_num, 3)
        for i in tqdm(range(len(edges))):
            e = edges[i]
            edge = e[:,:2] #源IP，目的IP
            
            ips = le.fit_transform(edge.reshape(-1))
            self.ip_list.append(le.classes_.tolist())
            
            edge=ips.reshape(-1,2)
            n = len(le.classes_)
            A_out = torch.sparse_coo_tensor([edge[:,0],range(self.flow_num)], torch.ones(self.flow_num), size=[n, self.flow_num])
            A_in = torch.sparse_coo_tensor([edge[:,1],range(self.flow_num)], torch.ones(self.flow_num), size=[n, self.flow_num])
            adj = torch.sparse_coo_tensor(edge.T, torch.ones(self.flow_num),size=[n, n])
            self.A_list.append(adj)
            adj = self.normalize_sym(torch.eye(adj.shape[0])+adj)
            A_out=self.normalize_random(A_out.to_dense())
            A_in=self.normalize_random(A_in.to_dense())
            self.Aout_list.append(A_out.to_sparse())#源IP-Flow邻接矩阵：有向图
            self.Ain_list.append(A_in.to_sparse()) #目的IP-Flow邻接矩阵: 有向图
            self.adj_list.append(adj.to_sparse()) # ip节点邻接矩阵：无向图
        
        # return 
        # return GraphData(self.Ain_list, self.Aout_list, self.A_list, self.adj_list)
        # return {
        #     'Ain_list':self.Ain_list,
        #     'Aout_list': self.Aout_list,
        #     'A_list':self.A_list,
        #     'adj_list':self.adj_list,
        #     'feat_list':self.feat_list,
        #     'ip_list': self.ip_list,
        #     'flow_list':self.flow_list
        # }

    def process_cst(self):
        # 读取文件
        dataframe_list=[]
        print("读取文件")
        for df in self.df_list:
            # df = df[~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)].dropna()
            print(f"读取文件{df.shape}")
            k = math.floor(len(df) / self.flow_num)
            df = df.iloc[:k * self.flow_num, :]
            dataframe_list.append(df.iloc[:k*self.flow_num,:])
        df = pd.concat(dataframe_list, ignore_index=True)
        self.df = df
        # Flow特征
        feats = df.drop(columns=[
            'ipv4_initiator', 'ipv4_responder', 'start_time'
        ])

        # 保存流量通信记录
        self.flow_list = df.loc[:, ['ipv4_initiator', 'ipv4_responder', 'start_time']]
        # df_record.to_csv(self.flow_file, index=False)
        
        # 特征归一化
        mm = MinMaxScaler()
        feats = mm.fit_transform(feats)
        print("生成特征列表")
        
        # 生成特征列表
        self.feat_list = feats.reshape(-1, self.flow_num, feats.shape[1])
        print("生成特征列表ok")
        # print(len(self.label_file))
        # print(len(self.edge_file))
        # print(len(self.feat_file))
        # 保存处理后的数据