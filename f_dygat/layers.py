import math

import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
# 衰减的时序聚合 
class GraphConvolution3(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution3, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lstm = nn.LSTM(32, 32,1)
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        '''为了让每次训练产生的初始参数尽可能相同，从而便于实验结果的复现，可以设置固定的随机数生成种子'''
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        self.lstm.reset_parameters()
    
    def decay(self, i, flow):
        if i==0:
            return flow[i]
        return flow[i]+0.5*self.decay(i-1,flow)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        # output = torch.spmm(adj, support) #稀疏矩阵相乘
        seq_node=[]
        aa = adj.to_dense()
        for a in aa:
            idx = torch.nonzero(a).squeeze(1)
            if len(idx)==0:
                seq_node.append(torch.zeros(self.out_features))
            else:
                idx = idx[len(idx)-100:len(idx)]
                seq_node.append(self.decay(len(idx)-1, support[idx]))
        output=torch.stack(seq_node)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        '''该方法是类的实例化对象用来做“自我介绍”的方法，默认情况下会返回当前对象的“类名+object at+内存地址”'''
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

# LSTM学习边的时序得到节点的特征d
class GraphConvolution2(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lstm = nn.LSTM(out_features, out_features, 1)
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        '''为了让每次训练产生的初始参数尽可能相同，从而便于实验结果的复现，可以设置固定的随机数生成种子'''
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        self.lstm.reset_parameters()

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        # output = torch.spmm(adj, support) #稀疏矩阵相乘
        seq_node=[]
        aa = adj.to_dense()
        for a in aa:
            idx = torch.nonzero(a).squeeze(1)
            seq_node.append(support[idx[len(idx)-10:len(idx)]]) # 选取后10个流输入到lstm中
        seq_node = pad_sequence(seq_node)
        output,_=self.lstm(seq_node)
        output=output[-1]
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        '''该方法是类的实例化对象用来做“自我介绍”的方法，默认情况下会返回当前对象的“类名+object at+内存地址”'''
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class FG(Module):

    def __init__(self, in_features, out_features, type=None):
        super(FG, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hiden_features = int((in_features + out_features) / 2)
        print(self.in_features, self.hiden_features, self.out_features)
        self.weight = Parameter(torch.FloatTensor(in_features, self.hiden_features))
        self.weight2=Parameter(torch.FloatTensor(self.hiden_features, out_features))
        self.layernorm= nn.LayerNorm(out_features)
        self.type=type
        self.reset_parameters()


    def reset_parameters(self):
        '''为了让每次训练产生的初始参数尽可能相同，从而便于实验结果的复现，可以设置固定的随机数生成种子'''
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.weight2.size(1))
        self.weight2.data.uniform_(-stdv, stdv)


    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj.to(support.device), support) #聚合邻居特征
        output=F.relu(output)
        output=torch.mm(output, self.weight2)
        output=self.layernorm(output)
        output=F.relu(output)
        return output

    def __repr__(self):
        '''该方法是类的实例化对象用来做“自我介绍”的方法，默认情况下会返回当前对象的“类名+object at+内存地址”'''
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


# 原始GCN层
class GraphConvolution(Module):

    def __init__(self, in_features, out_features, type=None):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, 32))
        self.weight2=Parameter(torch.FloatTensor(32, out_features))
        self.layernorm= nn.LayerNorm(out_features)
        self.type=type
        self.reset_parameters()


    def reset_parameters(self):
        '''为了让每次训练产生的初始参数尽可能相同，从而便于实验结果的复现，可以设置固定的随机数生成种子'''
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.weight2.size(1))
        self.weight2.data.uniform_(-stdv, stdv)


    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj.to(support.device), support) #聚合邻居特征
        output=F.relu(output)
        output=torch.mm(output, self.weight2)
        output=self.layernorm(output)
        output=F.relu(output)
        return output

    def __repr__(self):
        '''该方法是类的实例化对象用来做“自我介绍”的方法，默认情况下会返回当前对象的“类名+object at+内存地址”'''
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903 
    图注意力层
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features   # 节点表示向量的输入特征维度
        self.out_features = out_features   # 节点表示向量的输出特征维度
        self.dropout = dropout    # dropout参数
        self.alpha = alpha     # leakyrelu激活的参数
        self.concat = concat   # 如果为true, 再进行elu激活
        
        # 定义可训练参数，即论文中的W和a
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))  
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # xavier初始化
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)   # xavier初始化
        
        # 定义leakyrelu激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, inp, adj):
        """
        inp: input_fea [N, in_features]  in_features表示节点的输入特征向量元素个数
        adj: 图的邻接矩阵 维度[N, N] 非零即一，数据结构基本知识
        """
        h = torch.mm(inp, self.W)   # [N, out_features]
        N = h.size()[0]    # N 图的节点数
        
        a_input = torch.cat([h.repeat(1, N).view(N*N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2*self.out_features)
        # [N, N, 2*out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        # [N, N, 1] => [N, N] 图注意力的相关系数（未归一化）
        
        zero_vec = -1e12 * torch.ones_like(e)    # 将没有连接的边置为负无穷
        attention = torch.where(adj>0, e, zero_vec)   # [N, N]
        # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留，
        # 否则需要mask并置为非常小的值，原因是softmax的时候这个最小值会不考虑。
        attention = F.softmax(attention, dim=1)    # softmax形状保持不变 [N, N]，得到归一化的注意力权重！
        attention = F.dropout(attention, self.dropout, training=self.training)   # dropout，防止过拟合
        h_prime = torch.matmul(attention, h)  # [N, N].[N, out_features] => [N, out_features]
        # 得到由周围节点通过注意力权重进行更新的表示
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime 
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'