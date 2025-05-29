import numpy as np
import torch
import modellist
import torch.nn as nn
import torch.optim as optim
from tools import calratio
import tools
import matplotlib.pyplot as plt
import time
import random
torch.set_printoptions(precision=4) 
def get_sample_distance(points, labels):
    points = np.asarray(points)  # 转换为numpy 数组（如果输入是其他格式）
    labels = np.asarray(labels)
    
    # 提取下标为labels中的点
    selected_points = points[labels]
    
    # 计算所有点对之间的距离
    num_points = selected_points.shape[0]
    distances = np.zeros((num_points, num_points))  # 用于存储点对之间的距离
    
    # 计算点对距离（使用欧几里得距离）
    for i in range(num_points):
        for j in range(i + 1, num_points):
            dist = np.linalg.norm(selected_points[i] - selected_points[j])
            distances[i, j] = dist
            distances[j, i] = dist  # 距离矩阵是对称的
    
    return distances
def get_distance(points, labels):
    # 确保输入是NumPy数组
    points = np.asarray(points)
    labels = np.asarray(labels)
    
    # 提取下标为labels中的值的点
    selected_points = points[labels]
    
    # 计算所选点与其他所有点之间的欧式距离
    distances = np.sqrt(((points - selected_points[:, np.newaxis])**2).sum(axis=2))
    
    return distances
def Block_tropical(points): #一种minibatch的实现方式 
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu') 
    N,_ = points.shape
      
    P1 = modellist.TANN(N,1).to(device)
    optimizer = optim.RMSprop(P1.parameters(), lr=0.001)
    group_size = 100

# 使用列表推导和切片操作分组
    groups = [list(range(i, min(i + group_size, N))) for i in range(0, N, group_size)]
    groups = np.asarray(groups)
    criterion = nn.MSELoss()
    epochs = 1
    num_samples = 50
    sampled_points = random.sample(range(0, N), num_samples)
    for epoch in range(epochs):
        iter = 0
        for group in groups:
            iter = iter + 1
            print(iter)
            optimizer.zero_grad()
            len1 = group.shape
            dist = get_distance(points,group)
            P1_output = P1(group)
            group_tensor = torch.tensor(group).to(device)
            P1_output[torch.arange(len1[0]), group_tensor] = 0
            dist_tensor = torch.tensor(dist).to(device)
            loss = criterion(P1_output, dist_tensor)
            loss.backward()        # 执行反向传播，计算梯度
            optimizer.step()  
            P1.weight1.clamp(min=0)   # 将权重限制为非负
        sample_result = P1.sample(sampled_points)
        standard = get_sample_distance(points,sampled_points)
        standard_tensor = torch.tensor(standard).to(device)
        res = torch.norm(standard_tensor-sample_result, p='fro')**2
        ans = torch.norm(standard_tensor, p='fro')**2
        print(res)
        print(ans)
        print("epoch",epoch,"ratio",res/ans) #计算结果
    return res/ans
def Tropical(input_matrix):
    print(input_matrix)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    N,_ = data_numpy.shape
    mean = np.mean(data_numpy)
    P1 = modellist.TANN(N,mean).to(device)
    data_tensor = torch.from_numpy(input_matrix)
    optimizer = optim.AdamW(P1.parameters(), lr=0.001)

    y = data_tensor.to(device)    
    print(data_tensor.shape)
    num_epoch = 100
    dist_history = []
    min_dist = 1000
    time1 = time.time()
    loss_fn = nn.MSELoss()
    for epoch in range(num_epoch):
        # 生成P1矩阵
        optimizer.zero_grad()
        P1_output = P1()
        mask = torch.triu(torch.ones(N, N), diagonal=0)
        mask2 = torch.ones(N, N) - torch.eye(N)
        mask = mask.to(device)
        mask2 = mask2.to(device)
        loss = torch.sum(mask2*(P1_output-y) ** 2)
        loss.backward()
        dist = calratio(mask2 * (P1_output - y), y)
        dist_history.append(dist.item())  # 确保你只保存标量值
        optimizer.step()
        if dist < min_dist:
            min_dist = dist
            min_anss = P1_output
        now = time.time()
        print(f'Epoch {epoch}, Loss: {loss.data}, Dist: {dist},MinDist:{min_dist},costtime:{now-time1}')
    time2 = time.time()
    costtime = time2-time1
    result = min_dist
    check_output = False
    plt.plot(range(num_epoch), dist_history, label='Dist')
    return costtime,result,check_output,min_anss

if __name__ == "__main__":
    # Create a sample dissimilarity matrix

    # Set the tolerance parameter

    N = 1000000
    points = torch.rand(N, 2)
    print(points.shape)
    time1 = time.time()
    ratio = Block_tropical(points)
    time2 = time.time()
    print("cost time",time2-time1 )
    print("result ",ratio)
#    print(P_output2.data)
#    print(data_numpy)
