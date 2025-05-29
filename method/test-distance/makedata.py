import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# 1. 欧式距离矩阵
def euclidean_distance_matrix(X):
    return np.linalg.norm(X[:, np.newaxis] - X, axis=2)

# 2. 曼哈顿距离矩阵
def manhattan_distance_matrix(X):
    return np.sum(np.abs(X[:, np.newaxis] - X), axis=2)

# 3. 汉明距离矩阵
def hamming_distance_matrix(X):
    return np.sum(X[:, np.newaxis] != X, axis=2)

# 4. 余弦距离矩阵
def cosine_distance_matrix(X):
    cosine_sim_matrix = cosine_similarity(X)
    return 1 - cosine_sim_matrix

def shortest_path_distance_matrix(n):
    # 创建一个连通图：使用最小生成树保证图的连通性
    G = nx.erdos_renyi_graph(n, 0.1)
    
    # 确保图是连通的，如果不是，使用生成最小生成树的方法
    while not nx.is_connected(G):
        G = nx.erdos_renyi_graph(n, 0.1)

    # 计算所有节点之间的最短路径
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            try:
                dist_matrix[i, j] = nx.shortest_path_length(G, source=i, target=j)
                dist_matrix[j, i] = dist_matrix[i, j]  # 对称矩阵
            except nx.NetworkXNoPath:
                dist_matrix[i, j] = dist_matrix[j, i] = np.inf  # 如果没有路径，则设为无穷大
                print("wow")
    return dist_matrix

# 随机数据生成函数
def generate_random_data(n, dim):
    return np.random.rand(n, dim)

# 保存生成的距离矩阵到文件
def save_distance_matrix(matrix, name, n, dim):
    os.makedirs('data', exist_ok=True)
    np.save(f"data/{name}_n{n}_dim{dim}.npy", matrix)
    print(f"Saved {name} for n={n}, dim={dim} to 'data/{name}_n{n}_dim{dim}.npy'")

# 设置参数
n_values = [100, 500, 1000]  # 选择不同的节点数量
dim = 10  # 设置每个点的维度

# 生成和保存各种距离矩阵
for n in n_values:
    print(f"Generating and saving distance matrices for n={n}, dim={dim}...")
    
    # 随机生成数据
    X = generate_random_data(n, dim)

    # 1. 欧式距离矩阵
    D_euclidean = euclidean_distance_matrix(X)
    save_distance_matrix(D_euclidean, "euclidean", n, dim)
    
    # 2. 曼哈顿距离矩阵
    D_manhattan = manhattan_distance_matrix(X)
    save_distance_matrix(D_manhattan, "manhattan", n, dim)
    
    # 3. 汉明距离矩阵
    # 需要将数据转换为二值化的矩阵
    X_bin = np.random.randint(0, 2, (n, dim))
    D_hamming = hamming_distance_matrix(X_bin)
    save_distance_matrix(D_hamming, "hamming", n, dim)
    
    # 4. 余弦距离矩阵
    D_cosine = cosine_distance_matrix(X)
    save_distance_matrix(D_cosine, "cosine", n, dim)
    
    # 5. 图最短路径距离矩阵
    D_shortest_path = shortest_path_distance_matrix(n)
    save_distance_matrix(D_shortest_path, "shortest_path", n, dim)

print("All distance matrices have been generated and saved successfully.")
