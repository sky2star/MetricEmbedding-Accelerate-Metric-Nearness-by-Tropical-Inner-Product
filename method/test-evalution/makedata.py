import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
#import tools
def generate_graph_t1(n):
    """
    Generate Graph-t1 with n nodes, where edge weights follow U(0, 1)
    """
    # Initialize the measurement matrix Do as a zero matrix of size n x n
    Do = np.zeros((n, n))
    
    # Assign random weights between 0 and 1 to each edge
    for i in range(n):
        for j in range(i+1, n):  # Only iterate over the upper triangle of the matrix (i < j)
            weight = np.random.uniform(0, 1)
            Do[i, j] = weight
            Do[j, i] = weight  # Since the graph is undirected, the weight is symmetric
    
    return Do

def generate_graph_t2(n):
    """
    Generate Graph-t2 with n nodes, where edge weights are given by ⌈1000 * u * v^2⌉
    with u ∼ U(0, 1) and v ∼ N(0, 1)
    """
    # Initialize the measurement matrix Do as a zero matrix of size n x n
    Do = np.zeros((n, n))
    
    # Assign custom weights based on u and v distributions
    for i in range(n):
        for j in range(i+1, n):  # Only iterate over the upper triangle of the matrix (i < j)
            u_random = np.random.uniform(0, 1)
            v_random = np.random.normal(0, 1)
            weight = np.ceil(1000 * u_random * v_random**2)
            Do[i, j] = weight
            Do[j, i] = weight  # Since the graph is undirected, the weight is symmetric
    
    return Do
def generate_graph_t3(num_vectors,vector_dim=40):
    """
    Generate Graph-t2 with n nodes, where edge weights are given by ⌈1000 * u * v^2⌉
    with u ∼ U(0, 1) and v ∼ N(0, 1)
    """
    # Initialize the measurement matrix Do as a zero matrix of size n x n
    vectors = np.random.rand(num_vectors, vector_dim)
    
    # Compute the cosine similarity matrix
    cos_sim_matrix = cosine_similarity(vectors)
    Do = 1-cos_sim_matrix
    np.fill_diagonal(Do, 0)
    return Do
# Create 'data' directory if it does not exist
os.makedirs('data', exist_ok=True)

n_values = [2000]

for n in n_values:
    # Generate and save Graph-t1
    print(f"Generating Graph-t1 with n = {n}")
    Do1 = generate_graph_t1(n)
    np.save(f"data/Graph_t1_n{n}.npy", Do1)
    print(Do1.shape)
    #print("check satisfy",tools.check(Do1))
    print(f"Saved Graph-t1 for n={n} to 'data/Graph_t1_n{n}.npy'")

    # Generate and save Graph-t2
    # print(f"\nGenerating Graph-t2 with n = {n}")
    # Do2 = generate_graph_t2(n)
    # np.save(f"data/Graph_t2_n{n}.npy", Do2)
    # #print("check satisfy",tools.check(Do2))
    # print(f"Saved Graph-t2 for n={n} to 'data/Graph_t2_n{n}.npy'")

    # print(f"\nGenerating Graph-t3 with n = {n}")
    # Do3 = generate_graph_t3(n)
    # np.save(f"data/Graph_t3_n{n}.npy", Do3)
    # #print("check satisfy",tools.check(Do2))
    # print(f"Saved Graph-t3 for n={n} to 'data/Graph_t3_n{n}.npy'")
