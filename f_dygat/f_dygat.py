from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import softmax
from torch.nn.functional import mse_loss
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from f_dygat.utils import set_seed, Namespace, plot_train_loss
from f_dygat.models import IPFeatExtractor, Classifier
from f_dygat.utils import FlowDataset, eval, get_edge_embs, get_edge_embs_dy
from f_dygat.read_data import Dataset
import pandas as pd
import os

class Flow_DyGAT():
    def __init__(self, config):
        # Parameters:
        self.device = config["device"]

        self.model_dir = config["model_dir"]
        f_dygat_model_dir = os.path.join(self.model_dir, "f_dygat")
        os.makedirs(f_dygat_model_dir, exist_ok=True)
        self.model_path = os.path.join(f_dygat_model_dir, 'model.pt')

        # self.output_dir = os.path.join(config["output_dir"], "f_dygat")
        # os.makedirs(self.output_dir, exist_ok=True)

        config = config["f_dygat"]
        self.in_dim = config["in_dim"]
        self.out_dim = config["out_dim"]
        self.learning_rate = config["learning_rate"]
        self.seq_len = config["seq_len"]
        self.epochs = config["epochs"]
        self.flow_num = config["flow_num"]


    def read_data(self, csv_list) -> Dataset:
        data = Dataset(csv_list, self.flow_num)
        t0=time()
        print('结构特征图生成时间为', time()-t0)
        ip_lens=[len(ips) for ips in data.ip_list]
        avg_graph_ip_count = sum(ip_lens)/len(ip_lens)
        print("平均每张图包含的节点数目", avg_graph_ip_count)
        return data, avg_graph_ip_count, len(set([ip for sublist in data.ip_list for ip in sublist]))

    def model_forward(self, data, i, seq_len, device, model, cls=None):
        x = data.feat_list[i: i+seq_len]
        x = th.FloatTensor(x).to(self.device)

        Ain = data.Ain_list[i: i+seq_len] # 目的IP-边: 入边有向图
        Aout = data.Aout_list[i: i+seq_len] # 源IP-边: 出边有向图
        adj = data.adj_list[i: i+seq_len] # 源IP-目的IP: 无向图
        ips = data.ip_list[i: i+seq_len] # 节点集

        cur_ips = data.ip_list[i+seq_len-1] # 当前时刻的图
        cur_A=data.A_list[i+seq_len-1] # 归一化前的IP邻接矩阵（当前时刻）
        node_feats = model(x, Ain, Aout, adj, ips, cur_ips)
        edge_embs, labels, pose_edges = get_edge_embs_dy(node_feats, cur_A, cls) # 边的正负采样
        return labels, edge_embs, pose_edges

    # LSTM 学习节点的时序性，若干个时刻的节点集作为一个序列
    def train(self, data):
        device=self.device
        model = IPFeatExtractor(self.in_dim, self.out_dim).to(device)
        cls = Classifier(indim=self.out_dim*2).to(device)
        opt_gcn = optim.Adam(model.parameters(),lr=self.learning_rate)
        opt_cls =  optim.Adam(cls.parameters(), lr=self.learning_rate)
        loss_gcn = nn.CrossEntropyLoss()
        seq_len=self.seq_len
        losses=[]
        model.train()
        cls.train()
        for epoch in range(self.epochs):
            gcn_loss=0
            for i in tqdm(range(0, len(data.feat_list)-seq_len)):
                opt_gcn.zero_grad()
                opt_cls.zero_grad()
                labels, edge_embs, pos_edges=self.model_forward(data, i, seq_len, device, model, cls)
                pred= cls(edge_embs) # 基于分类器计算边的异常分数
                loss = loss_gcn(pred, torch.LongTensor(labels).to(device=device))

                loss.backward()
                gcn_loss+=loss.item()
                opt_gcn.step()
                opt_cls.step()
            losses.append(gcn_loss/i)
            print('epoch:{:.4f}, gcn_loss:{:.4f}'.format(epoch, gcn_loss/i))
            torch.save({
                'gcn_model':model.state_dict(),
                'cls':cls.state_dict()
            }, self.model_path)
        plot_train_loss(losses)
        return losses

    def predict(self, data):
        seq_len=self.seq_len
        # if os.path.exists(graph_embs_path):
        #     graph_embs=torch.load(graph_embs_path)
        #     df_flow_with_preds = pd.read_csv(flow_with_preds_path)
        #     return graph_embs, df_flow_with_preds.values.reshape(-1,1000,5)[seq_len:] 

        device = self.device
        model = IPFeatExtractor(self.in_dim, self.out_dim).to(device)
        cls = Classifier(indim=self.out_dim*2).to(device)

        ck=torch.load(self.model_path)
        model.load_state_dict(ck['gcn_model'])
        cls.load_state_dict(ck['cls'])
        model.eval()
        cls.eval()
        graph_embs=[]
        data.feat_list=torch.FloatTensor(data.feat_list).to(device)
        predictions = [[0, 0] for _ in range(seq_len * 1000)]
        with torch.no_grad():
            for i in tqdm(range(len(data.feat_list)-seq_len)):
                _, _, pos_edges=self.model_forward(data,  i, seq_len, device, model, cls)
                pred= cls(pos_edges) # 基于分类器计算边的异常分数
                
                pred = softmax(pred,0)
                pos_edges=pos_edges*pred[:,0].unsqueeze(1)
                graph_embs.append(pos_edges.sum(0)) # 图的嵌入向量=边嵌入向量*边异常概率
                for j in range(pred.size(0)):
                    predictions.append([pred[j, 0].item(), pred[j, 1].item()])

        df_preds = pd.DataFrame(predictions, columns=['0_pred', '1_pred'])
        # graph_embs_path = os.path.join(self.output_dir, 'graph_embs.pt')
        # torch.save(torch.tensor(graph_embs), graph_embs_path)
        # print(f"图嵌入已保存到{graph_embs_path}")

        return np.array(graph_embs), df_preds
    
    def predict_with_flow_anomaly(self, data:Dataset):
        graph_embs, df_preds = self.predict(data)
        print(data.flow_list.shape)
        print(df_preds.shape)
        df_flow_with_preds = pd.concat([data.flow_list, df_preds], axis=1)
        return graph_embs, df_flow_with_preds.values.reshape(-1,1000,5)[self.seq_len:]

    def load_graph_embs(self):
        graph_embs_path = os.path.join(self.filepath, 'graph_embs.pt')
        flow_with_preds_path = os.path.join(self.filepath, 'flow_with_preds.pt')
        data_embs = torch.load(graph_embs_path)
        flow_with_preds = pd.read_csv(flow_with_preds_path).values.reshape(-1,1000,5)[self.seq_len:]
        return data_embs.numpy(), flow_with_preds