import numpy as np
from memory_profiler import profile
import time
import tools
def TRF(D, k):
    """
    Metric Nearness L2 Algorithm
    
    Args:
    D (np.ndarray): Dissimilarity matrix (n x n).
    k (float): Tolerance parameter for convergence.
    
    Returns:
    np.ndarray: The updated matrix M.
    """
    # Initialize primal variables (eij)
    time1 = time.time()
    n = D.shape[0]
    eij = np.zeros((n, n))
    # Initialize dual variables (zikj, zjki, zkij)
    zikj = np.zeros((n, n, n))
    # Initialize tolerance for convergence test
    delta = 1 + k
    maxit = 40
    # Convergence loop
    it = 0
    while delta > k :
        # Reset delta for this iteration
#        print("check output",tools.check(D + eij))
        M2 = D + eij
        result2 = tools.calratio(M2-D,D)
#        if it>=5:
#            print("check output",tools.check(M2))
        print(result2)
#        print("check output",tools.calration(D + eij-))
#        print(delta,it)
        print(time.time()-time1)
        print(delta,it)
#        print(time.time()-time1)
        delta = 0
        it = it + 1
        # Iterate over all triangles (i, k, j)
        for i in range(n):
            for k in range(i + 1, n):
                for j in range(k + 1, n):
                    # Compute the violation of triangle inequality
                    v = D[i, k] + D[k, j] - D[i, j]
                    
                    # Update the dual variables and primal variables
                    theta_star = 1 / 3 * (eij[i, j] - eij[i, k] - eij[k, j] - v)
                    theta = max(theta_star, -zikj[i, k, j])
                    
                    # Update primal variables (eij, eik, ekj)
                    eij[i, j] -= theta
                    eij[i, k] += theta
                    eij[k, j] += theta
                    
                    eij[j, i] -= theta
                    eij[k, i] += theta
                    eij[j, k] += theta
                    # Update dual variable zikj
                    zikj[i, k, j] += theta

                    # Accumulate changes in eij for convergence test
                    delta += abs(theta)
        
        # Check for convergence
        if delta <= k:
            break
        if it >= maxit:
            break
    # Return the updated matrix M
    M = D + eij
    return M
# Example usage
if __name__ == "__main__":
    # Create a sample dissimilarity matrix
    print("hello world")
    D = np.load("data/Graph_t1_n100.npy")
    # Set the tolerance parameter
    k = 1e-4
    
    # Run the TRF function
    time1 = time.time()
    M = TRF(D, k)
    time2 = time.time()
    result = tools.calratio(M-D,D)
    print("used time",time2-time1)
    print("result",result)    
#    print("Updated matrix M:\n", M)
    print("check output",tools.check(M))