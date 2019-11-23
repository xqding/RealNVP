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
import argparse

## pass an index to specify which potential to use
parser = argparse.ArgumentParser()
parser.add_argument("--potential_index", type = int, choices = range(1,5), required = True)

args = parser.parse_args()
potential_index = args.potential_index
U = [compute_U1, compute_U2, compute_U3, compute_U4]
compute_U = U[potential_index - 1]

## Masks used to define the number and the type of affine coupling layers
## In each mask, 1 means that the variable at the correspoding position is
## kept fixed in the affine couling layer
masks = [[1.0, 0.0],
         [0.0, 1.0],
         [1.0, 0.0],         
         [0.0, 1.0],
         [1.0, 0.0],         
         [0.0, 1.0],
         [1.0, 0.0],
         [0.0, 1.0]]

## dimenstion of hidden units used in scale and translation transformation
hidden_dim = 128

## construct the RealNVP_2D object
realNVP = RealNVP_2D(masks, hidden_dim)
if torch.cuda.device_count():
    realNVP = realNVP.cuda()
    
optimizer = optim.Adam(realNVP.parameters(), lr = 0.0001)
num_steps = 5000

## the following loop learns the RealNVP_2D model by potential energy
## defined in the ./script/functions.py
for idx_step in range(num_steps):
    Z = torch.normal(0, 1, size = (1024, 2))
    Z = Z.cuda()
    X, logdet = realNVP(Z)

    logp = -compute_U(X)
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
z = z.cpu().detach().numpy()

with open(f"./output/learn_by_potential_x_transformed_from_z_{potential_index}.pkl", 'wb') as file_handle:
    pickle.dump({'z': z, 'x': x}, file_handle)    
    
fig = plt.figure(0)
fig.clf()
plt.plot(x[:,0], x[:,1], ".")
plt.xlabel(r'$X_1$')
plt.ylabel(r'$X_2$')
plt.xlim([-4,4])
plt.ylim([-4,4])
plt.savefig(f"./output/learn_by_potential_x_transformed_from_z_{potential_index}.png")



