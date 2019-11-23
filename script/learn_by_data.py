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
device = next(realNVP.parameters()).device

optimizer = optim.Adam(realNVP.parameters(), lr = 0.0001)
num_steps = 5000

## the following loop learns the RealNVP_2D model by data
## in each loop, data is dynamically sampled from the scipy moon dataset
for idx_step in range(num_steps):
    ## sample data from the scipy moon dataset
    X, label = datasets.make_moons(n_samples = 512, noise = 0.05)
    X = torch.Tensor(X).to(device = device)

    ## transform data X to latent space Z
    z, logdet = realNVP.inverse(X)

    ## calculate the negative loglikelihood of X
    loss = torch.log(z.new_tensor([2*math.pi])) + torch.mean(torch.sum(0.5*z**2, -1) - logdet)
    
    optimizer.zero_grad()
    loss.backward()
    
    optimizer.step()

    if (idx_step + 1) % 100 == 0:
        print(f"idx_steps: {idx_step:}, loss: {loss.item():.5f}")
        
## after learning, we can test if the model can transform
## the moon data distribution into the normal distribution
X, label = datasets.make_moons(n_samples = 1000, noise = 0.05)
X = torch.Tensor(X).to(device = device)
z, logdet_jacobian = realNVP.inverse(X)
z = z.cpu().detach().numpy()

X = X.cpu().detach().numpy()
fig = plt.figure(2, figsize = (12.8, 4.8))
fig.clf()
plt.subplot(1,2,1)
plt.plot(X[label==0,0], X[label==0,1], ".")
plt.plot(X[label==1,0], X[label==1,1], ".")
plt.title("X sampled from Moon dataset")
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")

plt.subplot(1,2,2)
plt.plot(z[label==0,0], z[label==0,1], ".")
plt.plot(z[label==1,0], z[label==1,1], ".")
plt.title("Z transformed from X")
plt.xlabel(r"$z_1$")
plt.ylabel(r"$z_2$")
plt.savefig("./output/moon_z_transformed_from_x.png")

## after learning, we can also test if the model can transform
## the normal distribution into the moon data distribution 
z = torch.normal(0, 1, size = (1000, 2)).to(device = device)
X, _ = realNVP(z)
X = X.cpu().detach().numpy()
z = z.cpu().detach().numpy()

fig = plt.figure(2, figsize = (12.8, 4.8))
fig.clf()
plt.subplot(1,2,1)
plt.plot(z[:,0], z[:,1], ".")
plt.title("Z sampled from normal distribution")
plt.xlabel(r"$z_1$")
plt.ylabel(r"$z_2$")

plt.subplot(1,2,2)
plt.plot(X[:,0], X[:,1], ".")
plt.title("X transformed from Z")
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.savefig("./output/moon_x_transformed_from_z.png")

