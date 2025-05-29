import matplotlib.pyplot as plt
import random
import copy
import networkx as nx     #导入networkx包
import random			  #导入random包
import matplotlib
import numpy as np
import torch 
import torch.optim as optim
import copy
import tools
import time

import numpy as np

def embedding_calibration(D, mu=0.02):
    """
    Calibrate a distance matrix by an embedding calibration method.

    Parameters:
        D (numpy.ndarray): Pairwise distance matrix.
        mu (float): Default is 0.02.

    Returns:
        numpy.ndarray: Calibrated distance matrix.
    """
    gamma = -mu / np.max(D)
    low = np.exp(-mu)
    K = nearpsd(np.exp(gamma * D), maxits=10, low=low)
    C = np.log(K) / gamma
    np.fill_diagonal(C, 0)
    return (C + C.T) / 2

def nearpsd(A, maxits=100, low=0, high=1, dv=1):
    """
    Computes the nearest positive semi-definite matrix for a given square matrix.

    Parameters:
        A (numpy.ndarray): A square matrix to be calibrated.
        maxits (int): Maximum number of iterations allowed. Default is 100.
        low (float): Minimum value threshold. Default is 0.
        high (float): Maximum value threshold. Default is 1.
        dv (float): Values of the diagonal elements. Default is 1.

    Returns:
        numpy.ndarray: Nearest positive semi-definite matrix to A.
    """
    if not np.allclose(A, A.T):
        A = (A + A.T) / 2

    tolconv = 1.0e-6
    toleigs = 1.0e-5

    n = A.shape[0]
    U = np.zeros_like(A)
    Y = A.copy()

    for _ in range(maxits):
        T = Y - U

        # Project onto PSD matrices
        eigvals, eigvecs = np.linalg.eigh(T)
        eigvals = np.maximum(eigvals, toleigs * eigvals[-1])
        X = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # Update correction
        U = X - T

        # Convergence test
        if np.linalg.norm(Y - X, ord=np.inf) / np.linalg.norm(Y, ord=np.inf) <= tolconv:
            break

        # Update Y with problem-specific constraints
        Y = X
        np.fill_diagonal(Y, dv)
        Y = np.clip(Y, low, high)

    np.fill_diagonal(Y, dv)
    Y = np.clip(Y, low, high)

    return Y

def heuristic_improve(X0, D, niters):
    """
    Perform an iterative matrix update based on the given rules.

    Parameters:
        X0 (numpy.ndarray): Initial matrix (n x n).
        D (numpy.ndarray): Distance matrix (n x n).
        niters (int): Number of iterations.

    Returns:
        numpy.ndarray: Updated matrix after iterations.
    """
    X = X0.copy()
    n = X.shape[0]

    for _ in range(niters):
        for i in range(n):
            for j in range(i + 1, n):
                if X[i, j] > D[i, j]:
                    X[i, j] = D[i, j]
                    X[j, i] = D[i, j]
                    for k in range(n):
                        v = abs(X[i, k] - X[j, k])
                        if v > X[i, j]:
                            X[i, j] = v
                else:
                    X[i, j] = D[i, j]
                    X[j, i] = D[i, j]
                    for k in range(n):
                        v = X[i, k] + X[j, k]
                        if v < X[i, j]:
                            X[i, j] = v
                X[j, i] = X[i, j]

    return X

def addm(D, X, lambda_, n):
    """
    Perform the operation: X = lambda * D + (1 - lambda) * X.

    Parameters:
        D (numpy.ndarray): Distance matrix.
        X (numpy.ndarray): Current matrix.
        lambda_ (float): Weight parameter.
        n (int): Size of the matrix.
    """
    lambda2 = 1.0 - lambda_
    X[:] = lambda_ * D + lambda2 * X

def compute_nmse(D, X):
    """
    Compute the normalized mean square error (NMSE) between two matrices.

    Parameters:
        D (numpy.ndarray): Distance matrix.
        X (numpy.ndarray): Current matrix.

    Returns:
        float: NMSE value.
    """
    se = np.sum((D - X) ** 2)
    normD = np.sum(D ** 2)
    print(se,normD)
    return se / normD

def compute_vltns(X):
    """
    Compute the number of violations to the triangle inequality in the matrix.

    Parameters:
        X (numpy.ndarray): Current matrix.

    Returns:
        float: Number of violations.
    """
    n = X.shape[0]
    count = 0.0
    epsilon = 1.0e-10
    for i in range(n):
        for j in range(i + 1, n):
            ab = X[i, j]
            for k in range(n):
                if k == i or k == j:
                    continue
                if ab - epsilon > X[i, k] + X[k, j]:
                    count += 1.0
    return count

def hlwb_projection(X0, D0, niters,start_time):
    """
    Perform the iterative matrix update process.

    Parameters:
        X0 (numpy.ndarray): Initial matrix (n x n).
        D0 (numpy.ndarray): Distance matrix (n x n).
        niters (int): Number of iterations.

    Returns:
        numpy.ndarray: Updated matrix after iterations.
    """
    X = X0.copy()
    n = X.shape[0]

    print(f"Data matrix: {n} x {n}")
    print("HLWB projection entered")
    print("Iters\tNMSE_before\tVLTN_before\tUpdates\tNMSE_after\tVLTN_after\tSecs")

    updates = 0.0
    for iter in range(1, niters + 1):
        addm(D0, X, 0.382 / iter, n)
        nmse_before = compute_nmse(D0, X)
        vltn_before = compute_vltns(X) / 1.0e3
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    delta = X[i, j] - X[i, k] - X[j, k]
                    if delta > 0.0:
                        delta /= 3.0
                        X[i, j] -= delta
                        X[j, i] = X[i, j]
                        X[i, k] += delta
                        X[k, i] = X[i, k]
                        X[j, k] += delta
                        X[k, j] = X[j, k]
                        updates += 3.0
                        continue

                    delta = X[i, k] - X[i, j] - X[j, k]
                    if delta > 0.0:
                        delta /= 3.0
                        X[i, k] -= delta
                        X[k, i] = X[i, k]
                        X[i, j] += delta
                        X[j, i] = X[i, j]
                        X[j, k] += delta
                        X[k, j] = X[j, k]
                        updates += 3.0
                        continue

                    delta = X[j, k] - X[i, j] - X[i, k]
                    if delta > 0.0:
                        delta /= 3.0
                        X[j, k] -= delta
                        X[k, j] = X[j, k]
                        X[i, j] += delta
                        X[j, i] = X[i, j]
                        X[i, k] += delta
                        X[k, i] = X[i, k]
                        updates += 3.0
                        continue

        elapsed_time = time.time() - start_time
        nmse_after = compute_nmse(D0, X)
        vltn_after = compute_vltns(X) / 1.0e3

        print(f"{iter}\t{nmse_before:.7f}\t{vltn_before:.3f}K\t{updates / 1.0e6:.6f}M\t{nmse_after:.7f}\t{vltn_after:.3f}K\t{elapsed_time:.3f}")
    # Zero diagonal and make symmetric
    np.fill_diagonal(X, 0.0)
#    X = (X + X.T) / 2.0
    return X
def HLWBopt(I):
    maxit = 50
    n,_ = I.shape
    X=copy.deepcopy(I)
    D=copy.deepcopy(I)
    loss_list=[]
    #itlist=[]
    num=0
    time1=time.time()
    for t in range(maxit):
        k=t
        Xt=copy.deepcopy(X)
        X=1/(t+2)*D+(t+1)/(t+2)*X
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if i==j or j==k or k==i :
                        continue
                    delta=(X[i][j]-X[i][k]-X[k][j])/3
                    if delta>0:
                        X[i][j]=X[i][j]-delta
                        X[i][k]=X[i][k]+delta
                        X[k][j]=X[k][j]+delta
                        num=num+1
        
        norm = np.linalg.norm(X-D, ord=2) 
#        print(num,norm)
        loss_list.append(norm)
#
# 
# if norm<0.001 :
#            break
    time2 = time.time()
    plt.plot(loss_list)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Plotting  Data')
    plt.show()
    plt.savefig("loss-HLWB.png")
    print("time result",time2-time1)
    print("ratio",tools.calratio(X-I,I))
    return X
def opt(Dnoise):
    time1=time.time()
    Dcal = embedding_calibration(Dnoise)
    Dheu = heuristic_improve(Dcal, Dnoise, 1)
    Dhlwb = hlwb_projection(Dheu, Dnoise, 20,time1)
    time2=time.time()
    result = tools.calratio(Dhlwb-Dnoise,Dnoise)
    costtime = time2-time1
#    print("cost time",time2-time1)
#    print("ratio",result)
    checkoutput = tools.check(Dhlwb)
#    print("check",checkoutput)
    return costtime,result,checkoutput,Dhlwb

if __name__ == "__main__":
    # Create a sample dissimilarity matrix
#    Dnoise = np.load('test.npy')
    Dnoise = np.load('data/Graph_t1_n100.npy')
    time1=time.time()
    Dcal = embedding_calibration(Dnoise)
    Dheu = heuristic_improve(Dcal, Dnoise, 1)
    Dhlwb = hlwb_projection(Dheu, Dnoise, 40,time1)
    time2=time.time()
    result = tools.calratio(Dhlwb-Dnoise,Dnoise)
    print("cost time",time2-time1)
    print("ratio",result)
    checkoutput = tools.check(Dhlwb)
    print("check",checkoutput)
    #output = HLWBopt(data_numpy)
    # Run the TRF function
