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
from sklearn import datasets
from RealNVP_2D import *

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
num_steps = 10000

for idx_step in range(num_steps):
    X, label = datasets.make_moons(n_samples = 512, noise = 0.05)
    X = torch.Tensor(X)    
    X = X.cuda()
    
    z, logdet = realNVP.inverse(X)
    loss = torch.log(z.new_tensor([2*math.pi])) + torch.mean(torch.sum(0.5*z**2, -1) - logdet)
    
    optimizer.zero_grad()
    loss.backward()
    
    optimizer.step()

    if (idx_step + 1) % 100 == 0:
        print(f"idx_steps: {idx_step:}, loss: {loss.item():.5f}")

X, label = datasets.make_moons(n_samples = 1000, noise = 0.05)
X = torch.Tensor(X)
X = X.cuda()
z, logdet_jacobian = realNVP.inverse(X)
z = z.cpu().detach().numpy()

fig = plt.figure(0)
fig.clf()
plt.plot(z[label==0,0], z[label==0,1], ".")
plt.plot(z[label==1,0], z[label==1,1], ".")
plt.savefig("./output/moon_samples_z.pdf")

z = torch.normal(0, 1, size = (1000, 2))
z = z.cuda()
x, _ = realNVP(z)
x = x.cpu().detach().numpy()

fig = plt.figure(0)
fig.clf()
plt.plot(x[:,0], x[:,1], ".")
plt.savefig("./output/moon_samples_x.pdf")

