import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import MetricNN
import tools
import modellist
import time
#import makedata
# 读取数据函数
def compute_distances(X):
    N = X.shape[0]
    distances = np.zeros((N, N))  # 初始化一个N x N的矩阵来存储点对距离
    for i in range(N):
        for j in range(i + 1, N):  # 只计算上三角部分，因为距离矩阵是对称的
            dist = np.linalg.norm(X[i] - X[j])
            distances[i, j] = dist
            distances[j, i] = dist  # 对称赋值
    return distances
def add_noise(distances, noise_level):
    noise = np.random.normal(0, noise_level, distances.shape)
    distances_noisy = distances + noise
    return distances_noisy
def load_data(file_path):
    """从指定路径加载数据"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 未找到！")
    return np.load(file_path)

# 数据恢复函数
def restore_data(noisy_data, model, criterion, optimizer, mask,epochs=1000):
    """
    使用给定的模型根据噪声数据恢复完整的数据
    noisy_data: 带噪声的数据
    model: 用于恢复数据的神经网络模型
    criterion: 损失函数
    optimizer: 优化器
    epochs: 训练的轮次
    """
    model.train()
    noisy_data_tensor = torch.tensor(noisy_data, dtype=torch.float32)
    mask = torch.tensor(mask,dtype=torch.float32)
#    mask = torch.isnan(distances_missing)  # True where NaN, False where not NaN
#    mask = ~mask
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 通过模型恢复数据
        restored_data = model()
        restored_data.fill_diagonal_(0)
        # 计算损失
        loss = torch.sum(mask * (noisy_data_tensor - restored_data) ** 2)
        dist = tools.calratio(noisy_data_tensor - restored_data, noisy_data_tensor)
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
 #       if epoch % 100 == 0:  # 每100轮打印一次损失
        print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f},dist:{dist}")
    
    return restored_data.detach().numpy()

# 比较恢复结果与原始数据的函数
def compare_results(restored_data, ground_truth):
    """
    比较恢复的数据和真实的ground truth数据
    计算 ||A - B||_F^2 / ||B||_F^2 作为误差度量
    """
    # 计算Frobenius范数的平方
    frobenius_norm_diff_sq = np.sum((restored_data - ground_truth) ** 2)
    frobenius_norm_ground_truth_sq = np.sum(ground_truth ** 2)

    # 计算 ||A - B||_F^2 / ||B||_F^2
    relative_error = frobenius_norm_diff_sq / frobenius_norm_ground_truth_sq
    
    print(f"Relative Error (||A - B||_F^2 / ||B||_F^2): {relative_error:.4f}")
    return relative_error

# 主函数
def main():
    # 设置文件路径
    noisy_data_path = 'data/point_distances_missing_50.npy'
#    noisy_data_path = 'data/point_distances_noisy_30.npy'  # 示例噪声数据
    ground_truth_path = 'data/point_distances_original.npy'  # 原始数据

    # 加载数据
    '''
    N = 100
    X = np.random.rand(N, 2)
    ground_truth = compute_distances(X)
    noisy_data = add_noise(ground_truth,0.4)
    '''
    noisy_data = load_data(noisy_data_path)
    ground_truth = load_data(ground_truth_path)
    noisy_data_filled = np.nan_to_num(noisy_data, nan=0.0)
    mask = np.isnan(noisy_data)
    mask = ~mask
    dist1 = compare_results(noisy_data_filled, ground_truth)
    mean = np.mean(noisy_data_filled)
    # 初始化模型
    input_dim = noisy_data.shape[0]  # 假设数据为二维矩阵
#    output_dim = noisy_data.shape[1]  # 假设数据为二维矩阵
    model =MetricNN.MetricNN(input_dim,mean)
    # 设置损失函数和优化器
    criterion = nn.MSELoss()  # 使用均方误差作为损失函数
    optimizer = optim.RMSprop(model.parameters(), lr=0.01)
    
    # 恢复数据
    time1 = time.time()
    restored_data = restore_data(noisy_data_filled, model, criterion, optimizer,mask, epochs=100)
    time2 = time.time()
#    print(X)
#    print(noisy_data)
#    print(restored_data )
    # 比较恢复结果与ground truth
#    compare_results(noisy_data, ground_truth)
    dist2 = compare_results(restored_data, ground_truth)
    print(dist1,dist2,time2-time1,tools.check(restored_data))

# 执行主函数
if __name__ == "__main__":
    main()
