import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, to_tree, dendrogram
import matplotlib.pyplot as plt
# A helper class for KitNET which performs a correlation-based incremental clustering of the dimensions in X
# n: the number of dimensions in the dataset
# For more information and citation, please see our NDSS'18 paper: Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection
class corClust:
    def __init__(self,n):
        #parameter:
        self.n = n
        #varaibles
        self.c = np.zeros(n) #linear num of features
        self.c_r = np.zeros(n) #linear sum of feature residules
        self.c_rs = np.zeros(n) #linear sum of feature residules
        self.C = np.zeros((n,n)) #partial correlation matrix
        self.N = 0 #number of updates performed

    # x: a numpy vector of length n
    def update(self,x):
        self.N += 1
        self.c += x
        c_rt = x - self.c/self.N
        self.c_r += c_rt
        self.c_rs += c_rt**2
        self.C += np.outer(c_rt,c_rt)

    # creates the current correlation distance matrix between the features
    def corrDist(self):
        c_rs_sqrt = np.sqrt(self.c_rs)
        C_rs_sqrt = np.outer(c_rs_sqrt,c_rs_sqrt)
        C_rs_sqrt[C_rs_sqrt==0] = 1e-100 #this protects against dive by zero erros (occurs when a feature is a constant)
        D = 1-self.C/C_rs_sqrt #the correlation distance matrix
        D[D<0] = 0 #small negatives may appear due to the incremental fashion in which we update the mean. Therefore, we 'fix' them
        return D

    # clusters the features together, having no more than maxClust features per cluster
    def cluster(self,maxClust):
        D = self.corrDist()
        Z = linkage(D[np.triu_indices(self.n, 1)])  # create a linkage matrix based on the distance matrix
        plt.figure(figsize=(10, 7))
        dendrogram(Z)
        plt.title("Dendrogram of Feature Clustering")
        plt.xlabel("Feature Index")
        plt.ylabel("Distance")
        plt.savefig("tree.png")
        if maxClust < 1:
            maxClust = 1
        if maxClust > self.n:
            maxClust = self.n
        # # 初始自上而下的聚类
        # initial_clusters = self.__breakClust__(to_tree(Z), maxClust)
        # # 自下而上地补充特征
        # final_clusters = self.expand_clusters_bottom_up(initial_clusters, D, 5)
        # return final_clusters
        # 初始自上而下的聚类
        initial_clusters = self.__breakClust__(to_tree(Z), maxClust)
        return initial_clusters
    
    # a recursive helper function which breaks down the dendrogram branches until all clusters have no more than maxClust elements
    def __breakClust__(self,dendro,maxClust):
        if dendro.count <= maxClust: #base case: we found a minimal cluster, so mark it
            return [dendro.pre_order()] #return the origional ids of the features in this cluster
        return self.__breakClust__(dendro.get_left(),maxClust) + self.__breakClust__(dendro.get_right(),maxClust)

    def expand_clusters_bottom_up(self, clusters, D, minClust):
        # clusters: 初始聚类结果，列表的列表，每个列表包含特征索引
        # D: 相关性距离矩阵
        # minClust: 最小特征数量
        # 返回满足最小特征数量要求的簇列表

        feature_indices = set(range(self.n))
        # 创建特征到簇的映射，允许特征被分配到多个簇
        feature_to_clusters = {i: set() for i in feature_indices}
        for idx, cluster in enumerate(clusters):
            for feature in cluster:
                feature_to_clusters[feature].add(idx)

        # 转换簇为集合以便操作
        cluster_sets = [set(cluster) for cluster in clusters]

        # 自下而上地补充特征
        for idx, cluster in enumerate(cluster_sets):
            while len(cluster) < minClust:
                # 计算未在当前簇中的特征与簇的平均相关性距离
                remaining_features = feature_indices - cluster
                avg_distances = []
                for feature in remaining_features:
                    distances = [D[feature, c] for c in cluster]
                    avg_distance = np.mean(distances)
                    avg_distances.append((avg_distance, feature))
                # 按照平均距离从小到大排序（相关性从高到低）
                avg_distances.sort()
                if not avg_distances:
                    # 没有剩余特征可供选择，退出循环
                    break
                # 选择相关性最高的特征
                _, best_feature = avg_distances[0]
                # 将特征添加到簇中
                cluster.add(best_feature)
                # 更新特征到簇的映射
                feature_to_clusters[best_feature].add(idx)
        # 将集合转换回列表
        final_clusters = [list(cluster) for cluster in cluster_sets]
        return final_clusters

# Copyright (c) 2017 Yisroel Mirsky
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.