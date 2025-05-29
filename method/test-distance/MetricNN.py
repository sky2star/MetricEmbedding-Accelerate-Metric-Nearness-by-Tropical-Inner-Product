import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import torch.nn.functional as F
import torch.nn.init as init
class TropicalDotProduct(nn.Module):
    def __init__(self):
        super(TropicalDotProduct, self).__init__()

    def forward(self, x, w):
        """
        计算Tropical内积：max(x + w)
        参数:
        x (tensor): 输入矩阵，形状为 (N, K)
        w (tensor): 权重矩阵，形状为 (M, K)
        
        返回:
        tensor: 经过tropical内积的结果，形状为 (N, M)
        """
        N, K = x.shape
        M, _ = w.shape
        
        # 扩展输入x和权重w，形成N x M x K矩阵
        expanded_x = torch.abs(x).unsqueeze(1).expand(N, M, K)  # N x M x K
        expanded_w = torch.abs(w).unsqueeze(0).expand(N, M, K)  # N x M x K
        # 计算元素级别的和，并取每列的最大值
        sum_result = expanded_x + expanded_w
        max_result = torch.max(sum_result, dim=2)[0]  # 对每列取max
        
        return max_result
        
class TAlayer(nn.Module):
    def __init__(self, in_features, out_features,init_scale):
        super(TAlayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features)* init_scale) #K*M-> mk

    def forward(self, x):
        N = x.shape[0]
        K = x.shape[1]
        M = self.weight.shape[0]
        expanded_x = torch.abs(x).unsqueeze(1).expand(N, M, K) # N x M x K
        expanded_w = torch.abs(self.weight).unsqueeze(0).expand(N, M, K)  # N x M x K
        
        sum_result = expanded_x + expanded_w
        max_result = torch.max(sum_result, dim=2)[0]
        return max_result  
class MetricNN(nn.Module):
    def __init__(self, N, mean):
        """
        初始化MetricNN网络
        N: 矩阵维度
        k: 潜在空间维度
        init_scale: 初始化缩放因子
        """
        super(MetricNN, self).__init__()

        self.N = N
        k = 10
        d = 7
        # 定义多个TAlayer
        init_scale = mean / d /2
        self.x = nn.Parameter(torch.rand(N, 10)*init_scale)
#        self.layer1 = TAlayer(N, 256, init_scale)
        self.layer1 = TAlayer(10, 3, init_scale)
        self.b1 = nn.Parameter(torch.randn(20) * init_scale)
        self.layer2 = TAlayer(20, 3, init_scale)
        self.b2 = nn.Parameter(torch.randn(20) * init_scale)
        # 定义Tropical内积层
        self.tropical_dot = TropicalDotProduct()

    def forward(self):
        """
        前向传播，计算MetricNN输出
        """
        # 经过TAlayer的转换
        out = self.x
        out = self.layer1(out)
#        out = self.layer1(self.x)
#        out = torch.max(out, torch.abs(self.b1))
#        out = self.layer2(out)
#        out = torch.max(out, torch.abs(self.b2))
#        out2 =self.layer2(out1)
#        out1 = self.layer1(self.x)
#        out2 = out1
#        out1 = torch.max(out1, torch.abs(self.b1))
#        out2 = self.layer2(out1)
#        out2 = torch.max(out2, torch.abs(self.b2))
#        print(out2.shape,self.b1.shape)
#        out2 = torch.max(out2, self.b1)
#        out3 = self.layer3(out2)
        
        # 使用Tropical内积计算A和B的距离
        dist_AB = self.tropical_dot(out, out)
        return dist_AB

    def loss(self, dist_AB, true_dist):
        """
        定义损失函数，通常使用Frobenius norm或者其他度量来计算距离
        """
        loss = torch.norm(dist_AB - true_dist, p='fro')
        return loss