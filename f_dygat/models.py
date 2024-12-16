import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import Tanh
from torch.nn.parameter import Parameter
from torch.utils.data import dataset
from torch.nn.functional import softmax
import f_dygat.utils as u
from f_dygat.layers import FG
from torch_geometric.nn import GATConv

class IPFeatExtractor(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(IPFeatExtractor, self).__init__()
        self.in_dim=in_dim
        self.out_dim=out_dim
        self.gc1 = FG(self.in_dim, self.out_dim) # 聚合出边特征
        self.gc2 = FG(self.in_dim, self.out_dim) # 聚合入边特征
        self.gc3 = GATConv(self.out_dim * 2, self.out_dim * 2, concat=False)
        self.lstm=nn.LSTM(input_size=self.out_dim * 2, hidden_size=self.out_dim ) #学习节点的时序性
        self.dropout = 0.5
        self.ln=nn.LayerNorm(self.out_dim)
        
    def node_history(self, ips, cur_ips, output):
        '''根据当前时刻的节点集cur_ips,获取历史i时刻节点的属性'''
        idx1 = np.where(np.in1d(ips, cur_ips))[0]
        idx2 = np.where(np.in1d(cur_ips, ips))[0]
        aa = torch.zeros(len(cur_ips), self.out_dim * 2).to(output.device)
        aa[idx2]=output[idx1]
        return aa.unsqueeze(0)

    def forward(self, x_list, Ain_list, Aout_list, A_list, ips_list, cur_ips):
        '''
        x_list: 边的特征向量
        Ain_list: 入向邻接矩阵
        Aout_list: 出边邻接矩阵
        A_list: 节点邻接矩阵
        ips_list: 节点集序列
        cur_ips: 当前时刻节点集
        '''
        seqs=[]
        for i in range(len(Ain_list)):
            x, Ain, Aout, Adj = x_list[i], Ain_list[i], Aout_list[i], A_list[i]
            node_in= self.gc1(x, Ain) # 聚合节点的入边特征
            node_out = self.gc2(x, Aout) # 聚合节点的出边特征
            
            node_feat=torch.cat((node_in, node_out),1) # 拼接节点的入边特征-出边特征作为节点的聚合特征
            node_feat = self.gc3(node_feat, Adj)
            node_feat = F.dropout(node_feat, 0.5)
            seqs.append(self.node_history(ips_list[i], cur_ips, node_feat))
        seqs=torch.vstack(seqs)
        output, _ =self.lstm(seqs) # 学习节点的时序
        return output[-1]
    
class Classifier(torch.nn.Module):
    def __init__(self,indim,out_features=2):
        super(Classifier,self).__init__()
        activation = torch.nn.ReLU()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(in_features = indim, out_features = 16),
                                       activation,
                                       torch.nn.Linear(in_features = 16,out_features = out_features))

    def forward(self,x):
        return self.mlp(x)

