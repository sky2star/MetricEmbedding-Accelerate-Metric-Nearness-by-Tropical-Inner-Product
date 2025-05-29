import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import MetricNN
import tools
import time
# 读取数据函数
def compute_distances(X):
    N = X.shape[0]
    distances = np.zeros((N, N))  # 初始化一个N x N的矩阵来存储点对距离
    for i in range(N):
        for j in range(i + 1, N):  # 只计算上三角部分，因为距离矩阵是对称的
            dist = np.sum(np.abs(X[i] - X[j])) 
            #            dist = np.linalg.norm(X[i] - X[j])
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
#        print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f},dist:{dist}")
    print(dist)
    return restored_data.detach().numpy()

# 比较恢复结果与原始数据的函数
def compare_results(restored_data, ground_truth):
    """
    比较恢复的数据和真实的ground truth数据
    计算 ||A - B||_F^2 / ||B||_F^2 作为误差度量
    """
    # 计算Frobenius范数的平方
#    print(restored_data[0][0:10])
#    print( ground_truth[0][0:10])   
    frobenius_norm_diff_sq = np.sum((restored_data - ground_truth) ** 2)
    frobenius_norm_ground_truth_sq = np.sum(ground_truth ** 2)
#    print(frobenius_norm_diff_sq ,frobenius_norm_ground_truth_sq)

    # 计算 ||A - B||_F^2 / ||B||_F^2
    relative_error = frobenius_norm_diff_sq / frobenius_norm_ground_truth_sq
    
    print(f"Relative Error (||A - B||_F^2 / ||B||_F^2): {relative_error:.4f}")
    return relative_error
def initialize_mask(N, exposed_ratio):
    """
    初始化一个Mask矩阵，暴露指定比例的元素
    N: 矩阵的大小（N x N）
    exposed_ratio: 暴露元素的比例
    """
    # 初始化一个全零的矩阵（表示所有元素都缺失）
    mask = np.zeros((N, N))
    
    # 计算需要暴露的元素数量
    num_exposed = int(N * N * exposed_ratio)
    
    # 随机选择需要暴露的元素位置
    exposed_indices = np.random.choice(N * N, size=num_exposed, replace=False)
    
    # 更新mask矩阵，暴露选择的元素
    mask.flat[exposed_indices] = 1
    
    return mask
def update_mask(exposed_ratio, original_mask):
    """
    更新mask矩阵，根据暴露比例更新缺失位置为1（暴露元素）

    mask: 当前的mask矩阵
    exposed_ratio: 当前暴露比例
    original_mask: 原始的mask矩阵，记录初始时缺失的位置
    """
    N = original_mask.shape[0]
    mask = original_mask
    # 计算原始mask矩阵中为0的位置，即缺失的数据
    missing_indices = np.where(original_mask == 0)  
    num = N*N  # 初始缺失位置的数量

    # 计算需要暴露的元素个数
    num_exposed = int(num * exposed_ratio)

    # 随机选择需要暴露的缺失元素
    exposed_indices = np.random.choice(len(missing_indices[0]), size=num_exposed, replace=False)

    # 更新Mask矩阵
    for idx in exposed_indices:
        row, col = missing_indices[0][idx], missing_indices[1][idx]
        mask[row, col] = 1  # 将对应位置设为1，即暴露该元素
    return mask
# 主函数
def compute_exposed_ratio(mask):
    """
    计算mask中为1的数量占所有元素的比例
    """
    # 统计mask中为1的元素个数
    num_exposed = np.sum(mask == 1)
    
    # 统计mask的总元素个数
    total_elements = mask.size
    
    # 计算比例
    exposed_ratio = num_exposed / total_elements
    
    return exposed_ratio
def main():
    # 设置文件路径
    cluster_center = np.array([50, 50])  # 聚集的中心
    cluster_points = np.random.randn(50, 2) * 5 + cluster_center  # 50个点，聚集在中心附近
    outliers_1 = np.random.randn(10, 2) * 20 + np.array([200, 200])  # 10个点，远离中心
    outliers_2 = np.random.randn(5, 2) * 15 + np.array([300, 300])   # 5个点，进一步远离
    outliers_3 = np.random.randn(5, 2) * 10 + np.array([400, 400])   # 5个点，距离更远

    np.random.seed(42)
    points = np.vstack([cluster_points, outliers_1, outliers_2, outliers_3])
        # 合并这两部分点
#    points = np.vstack([cluster_points, outliers])
    
    # 计算这些点之间的欧几里得距离
    ground_truth = np.linalg.norm(points[:, np.newaxis] - points, axis=2)
    cloned_ground = ground_truth.copy()
    
    N,_ = ground_truth.shape
    exposed_ratio = 0.2
    mask = initialize_mask(N, exposed_ratio)
    noisy_data_filled = ground_truth * mask
#    dist1 = compare_results(noisy_data_filled, ground_truth)
    mean = np.mean(noisy_data_filled)
    # 初始化模型
    input_dim = noisy_data_filled.shape[0]  # 假设数据为二维矩阵
    output_dim = noisy_data_filled.shape[1]  # 假设数据为二维矩阵
    model =MetricNN.MetricNN(input_dim,mean)
    # 设置损失函数和优化器
    criterion = nn.MSELoss()  # 使用均方误差作为损失函数
    optimizer = optim.RMSprop(model.parameters(), lr=0.01)
    
    # 恢复数据
    time1 = time.time()
    restored_data = restore_data(noisy_data_filled, model, criterion, optimizer,mask, epochs=100)
    time2 = time.time()
    dist2 = compare_results(restored_data, cloned_ground)
    ratio = compute_exposed_ratio(mask)
    print(ratio,dist2,tools.check(restored_data))
#    print(ground_truth)
#    print(noisy_data_filled)
#    print(restored_data)

    while exposed_ratio <= 1:
        exposed_ratio = exposed_ratio + 0.1
        mask = update_mask(0.1,mask)
        noisy_data_filled = ground_truth * mask
        ratio = compute_exposed_ratio(mask)
        restored_data = restore_data(noisy_data_filled, model, criterion, optimizer,mask, epochs=100)
        dist2 = compare_results(restored_data, ground_truth)
        print(ratio,dist2,tools.check(restored_data))
        if  exposed_ratio +0.1>1:
            break
# 执行主函数
if __name__ == "__main__":
    main()
