import numpy as np
import torch
import modellist
import torch.nn as nn
import torch.optim as optim
from tools import calratio
import matplotlib.pyplot as plt
import time
import HLWBtest
torch.set_printoptions(precision=4) 
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
    optimizer = optim.AdamW(P1.parameters(), lr=0.01)
    y = data_tensor.to(device)  
    print(data_tensor.shape)
    num_epoch = 200
    dist_history = []
    min_dist = 100
    time1 = time.time()
    loss_fn = nn.MSELoss()
    zeros = torch.zeros_like(P1.weight1)
    mask2 = torch.ones(N, N) - torch.eye(N)
    mask2 = mask2.to(device)
    for epoch in range(num_epoch):
        # 生成P1矩阵
        optimizer.zero_grad()
        P1_output = P1()
        loss = torch.sum(mask2*(P1_output-y) ** 2)
        loss.backward()
        dist = calratio(mask2 * (P1_output - y), y)
        dist_history.append(dist.item())  # 确保你只保存标量值
#        print(loss,dist)
        optimizer.step()
        P1.weight1.data = P1.weight1.data.clamp(min=0)
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

    data_numpy = np.load('data/Graph_t1_n100.npy')
    data_numpy = data_numpy
#    costtime,result,check_output,P_output = HLWBtest.opt(data_numpy)
    dist_history=[]
#    original = tools.check(data_numpy)
    costtime,result,check_output,P_output = Tropical(data_numpy)
    min_dist=100
    # for k in range(1,100,1):
    #     costtime,result,check_output,P_output2 = Tropical_withk(data_numpy,k+1)
    #     dist_history.append(result)
    #     if result<min_dist:
    #         min_dist=result
    #     print("k",k,result,min_dist)
#    plt.plot(range(1,100,1), dist_history, label='Dist')
#    plt.savefig("dist.png")
#    print(original)
#     print("used time",costtime)
#     print("result",result)    
# #    print("Updated matrix M:\n", M)
#     print("check output",check_output)
#    print(original)
    print("used time",costtime)
    print("result",result)    
    print("check output",check_output)