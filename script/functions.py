__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2019/11/04 17:27:27"

import numpy as np
import torch
import math
import matplotlib.pyplot as plt

'''
Four 2-d potential functions, U(x), are defined.
The corresponding probability densities are defined as \log P(x) \propto \exp(-U(x))

'''

def compute_U1(z):
    mask = torch.ones_like(z)
    mask[:, 1] = 0.0

    U = (0.5*((torch.norm(z, dim = -1) - 2)/0.4)**2 - \
             torch.sum(mask*torch.log(torch.exp(-0.5*((z - 2)/0.6)**2) +
                                      torch.exp(-0.5*((z + 2)/0.6)**2)), -1))    
    return U

def compute_U2(z):
    w1 = torch.sin(2*math.pi*z[:,0]/4)
    U = 0.5*((z[:,1] - w1)/0.4)**2    
    return U

def compute_U3(z):
    w1 = torch.sin(2*math.pi*z[:,0]/4)
    w2 = 3*torch.exp(-0.5*((z[:,0] - 1)/0.6)**2)
    U = -torch.log(torch.exp(-0.5*((z[:,1] - w1)/0.35)**2) + torch.exp(-0.5*((z[:,1] - w1 + w2)/0.35)**2))
    return U
    
def compute_U4(z):
    w1 = torch.sin(2*math.pi*z[:,0]/4)
    w3 = 3*torch.sigmoid((z[:,0] - 1)/0.3)    
    U = -torch.log(torch.exp(-0.5*((z[:,1] - w1)/0.35)**2) + torch.exp(-0.5*((z[:,1] - w1 + w3)/0.35)**2))
    return U

def _generate_grid():
    z1 = torch.linspace(-4, 4)
    z2 = torch.linspace(-4, 4)
    grid_z1, grid_z2 = torch.meshgrid(z1, z2)
    grid = torch.stack([grid_z1, grid_z2], dim = -1)
    z = grid.reshape((-1, 2))
    return z
    
def test_U():
    z = _generate_grid()
    
    fig = plt.figure(0, figsize = (12.8, 4.8))
    fig.clf()
    
    plt.subplot(1,4,1)
    U1 = compute_U1(z)
    U1 = U1.reshape(100, 100)    
    p1 = torch.exp(-U1)
    plt.imshow(p1.T, origin = "lower", extent = (-4,4,-4,4))
    
    plt.subplot(1,4,2)
    U2 = compute_U2(z)
    U2 = U2.reshape(100, 100)    
    p2 = torch.exp(-U2)
    plt.imshow(p2.T, origin = "lower", extent = (-4,4,-4,4))    

    plt.subplot(1,4,3)
    U3 = compute_U3(z)
    U3 = U3.reshape(100, 100)    
    p3 = torch.exp(-U3)
    plt.imshow(p3.T, origin = "lower", extent = (-4,4,-4,4))    
    
    plt.subplot(1,4,4)
    U4 = compute_U4(z)
    U4 = U4.reshape(100, 100)    
    p4 = torch.exp(-U4)
    plt.imshow(p4.T, origin = "lower", extent = (-4,4,-4,4))    
    
    plt.savefig("./output/true_prop.png")
    
if __name__ == "__main__":
    test_U()
