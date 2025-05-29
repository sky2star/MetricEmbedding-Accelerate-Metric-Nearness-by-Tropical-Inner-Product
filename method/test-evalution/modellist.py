import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import torch.nn.functional as F
#用于检查模型内部状态
def print_trainable_parameters(model):
    print("Trainable parameters in the model:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Name: {name}, Shape: {param.shape}")
def print_all_parameters(model):
    print("All parameters in the model (including non-trainable):")
    for name, param in model.named_parameters():
        print(f"Name: {name}")
        print(f"  - Trainable: {param.requires_grad}")
        print(f"  - Shape: {param.shape}")
        print(f"  - Data type: {param.dtype}")
        print(f"  - Device: {param.device}\n")
def check_negative_params(module, name=""):
    for param_name, param in module.named_parameters():
        print(f"Name: {name}")
        print(f"  - Trainable: {param.requires_grad}")
        print(f"  - Shape: {param.shape}")
        print(f"  - Data type: {param.dtype}")
        print(f"  - Device: {param.device}\n")
        if (param.data < 0).any():
            print(f"Negative values found in {name}.{param_name}")
            # 打印负数的数量
            print(f"Number of negative elements: {(param.data < 0).sum().item()}")
        else:
            print(f"No negative values in {name}.{param_name}")
class Maxlayer(nn.Module):
    def __init__(self, in_features, out_features,init_scale):
        super(Maxlayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features)* init_scale) #K*M-> mk
        
#        self.bias = nn.Parameter(torch.rand(out_features))  在实践中,为了优化方便，可以选择忽略掉bias项
        
    def forward(self, x):
        N = x.shape[0]
        K = x.shape[1]
        M = self.weight.shape[0]      
        expanded_x = torch.abs(x.unsqueeze(1).expand(N, M, K))  # N x M x K
        expanded_w = torch.abs(self.weight.unsqueeze(0).expand(N, M, K))  # N x M x K
        
        sum_result = expanded_x + expanded_w
        max_result = torch.max(sum_result, dim=2)[0]
        return max_result  
class TANN(nn.Module):
    def __init__(self, N,mean):
        """
        初始化网络
        N: 矩阵维度
        k: 潜在空间维度
        init_scale: 初始化缩放因子
        """
        super().__init__()
        self.N = N
        
        # 将X参数化为网络的可学习参数
        self.weight1 = nn.Parameter(torch.rand(N, 10000)*mean/2)
#        self.weight2 = nn.Parameter(torch.randn(N, 40)*mean/2)
    def forward(self):
        """
        前向传播，计算H矩阵
        """
        N = self.weight1.shape[0]
        K = self.weight1.shape[1]
        M = self.weight1.shape[0]    
        expanded_x = torch.abs(self.weight1).unsqueeze(1).expand(N, M, K)  # N x M x K
        expanded_w = torch.abs(self.weight1).unsqueeze(0).expand(N, M, K)  # N x M x K
        
        sum_result = expanded_x + expanded_w
        max_result = torch.max(sum_result, dim=2)[0]
        
        return max_result 
        
class TANetwork(nn.Module):
    def __init__(self, N,input = 0.5):
        """
        初始化网络
        N: 矩阵维度
        k: 潜在空间维度
        init_scale: 初始化缩放因子
        """
        super().__init__()
        self.N = N
        self.k = N/10
        
        # 将X参数化为网络的可学习参数
        self.input = nn.Parameter(torch.randn(N, 40) * 1/4*0.1)
        self.layer1 = Maxlayer(40,40,1/4*0.1)
#        self.layer2 = Maxlayer(40,40)
#        self.layer3 = Maxlayer(40,40)
    def forward(self):
        """
        前向传播，计算H矩阵
        """
        x = self.layer1(self.input)
#        x = self.layer2(x)
#        x = self.layer3(x)
        N = x.shape[0]
        K = x.shape[1]
        M = x.shape[0]    
        expanded_x = x.unsqueeze(1).expand(N, M, K)  # N x M x K
        expanded_w = x.unsqueeze(0).expand(N, M, K)  # N x M x K
        
        sum_result = expanded_x + expanded_w
        max_result = torch.max(sum_result, dim=2)[0]
        
        return max_result 