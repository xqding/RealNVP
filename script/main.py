__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2019/11/03 21:50:25"

import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import torch.optim as optim
import pickle
import math
from sys import exit
import matplotlib.pyplot as plt
from RealNVP_2D import *
from functions import *

masks = [[1.0, 0.0],
         [0.0, 1.0],
         [1.0, 0.0],         
         [0.0, 1.0],
         [1.0, 0.0],         
         [0.0, 1.0],
         [1.0, 0.0],
         [0.0, 1.0]]

hidden_dim = 128
realNVP = RealNVP_2D(masks, hidden_dim)
realNVP = realNVP.cuda()
optimizer = optim.Adam(realNVP.parameters(), lr = 0.0001)
num_steps = 5000

for idx_step in range(num_steps):
    Z = torch.normal(0, 1, size = (1024, 2))
    Z = Z.cuda()
    X, logdet = realNVP(Z)

    logp = compute_logp4(X)
    loss = torch.mean(-logdet - logp)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (idx_step + 1) % 100 == 0:
        print(f"idx_steps: {idx_step:}, loss: {loss.item():.5f}")

z = torch.normal(0, 1, size = (1000, 2))
z = z.cuda()
x, _ = realNVP(z)
x = x.cpu().detach().numpy()

fig = plt.figure(0)
fig.clf()
plt.plot(x[:,0], x[:,1], ".")
# plt.xlim([-4,4])
# plt.ylim([-4,4])
plt.savefig("./output/samples_x.pdf")

