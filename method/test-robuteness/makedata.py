import numpy as np
import os
import tools
# 创建数据保存文件夹
folder_path = 'data'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# 生成随机数据点，假设我们生成500个点，二维坐标
N = 100
X = np.random.rand(N, 2)

# 计算点对之间的欧几里得距离
def compute_distances(X):
    N = X.shape[0]
    distances = np.zeros((N, N))  # 初始化一个N x N的矩阵来存储点对距离
    for i in range(N):
        for j in range(i + 1, N):  # 只计算上三角部分，因为距离矩阵是对称的
            dist = np.linalg.norm(X[i] - X[j])
            distances[i, j] = dist
            distances[j, i] = dist  # 对称赋值
    return distances

# 生成原始点对距离矩阵
distances = compute_distances(X)
mean = np.mean(distances)
print(tools.check(distances))
# 设置缺失数据比例
missing_ratios = [0.1,0.2,0.3,0.4,0.5]  # 不同缺失率
noise_levels = [0.1,0.2,0.3,0.4,0.5]  # 不同噪声比例

# 生成缺失数据
def generate_missing_data(distances, missing_ratio):
    distances_missing = distances.copy()
    N = distances.shape[0]
    # 随机选择需要缺失的元素
    missing_indices = np.random.choice(N * N, size=int(N * N * missing_ratio), replace=False)
    for idx in missing_indices:
        row = idx // N
        col = idx % N
        distances_missing[row, col] = np.nan
#        distances_missing[col, row] = np.nan  # 对称赋值
    return distances_missing

# 生成加噪声数据
def add_noise(distances, noise_level=0.1):
#    std_val = np.std(distances)
    noise = np.random.normal(0, noise_level, distances.shape)
    distances_noisy = distances + noise
    distances_noisy = np.maximum(distances_noisy, 0)
    return distances_noisy


# 保存数据
def save_data(data, filename):
    np.save(filename, data)
    print(f"Saved data to {filename}")

# 原始数据
save_data(distances, os.path.join(folder_path, 'point_distances_original.npy'))


# 生成并保存不同缺失率的数据
for missing_ratio in missing_ratios:
    missing_data = generate_missing_data(distances, missing_ratio)
    save_data(missing_data, os.path.join(folder_path, f'point_distances_missing_{int(missing_ratio * 100)}.npy'))
    print(tools.check(missing_data))
# 生成并保存不同噪声比例的数据
for noise_level in noise_levels:
    noisy_data = add_noise(distances, noise_level)
    save_data(noisy_data, os.path.join(folder_path, f'point_distances_noisy_{int(noise_level * 100)}.npy'))
    print(tools.check(noisy_data))
