__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2019/11/04 17:27:27"

import numpy as np
import torch
import math
import matplotlib.pyplot as plt

def compute_logp1(z):
    mask = torch.ones_like(z)
    mask[:, 1] = 0.0

    logp = -(0.5*((torch.norm(z, dim = -1) - 2)/0.4)**2 - \
             torch.sum(mask*torch.log(torch.exp(-0.5*((z - 2)/0.6)**2) +
                                      torch.exp(-0.5*((z + 2)/0.6)**2)), -1))    
    return logp

def compute_logp2(z):
    w1 = torch.sin(2*math.pi*z[:,0]/4)
    logp = -0.5*((z[:,1] - w1)/0.4)**2    
    return logp

def compute_logp3(z):
    w1 = torch.sin(2*math.pi*z[:,0]/4)
    w2 = 3*torch.exp(-0.5*((z[:,0] - 1)/0.6)**2)
    logp = torch.log(torch.exp(-0.5*((z[:,1] - w1)/0.35)**2) + torch.exp(-0.5*((z[:,1] - w1 + w2)/0.35)**2))
    return logp
    
def compute_logp4(z):
    w1 = torch.sin(2*math.pi*z[:,0]/4)
    w3 = 3*torch.sigmoid((z[:,0] - 1)/0.3)    
    logp = torch.log(torch.exp(-0.5*((z[:,1] - w1)/0.35)**2) + torch.exp(-0.5*((z[:,1] - w1 + w3)/0.35)**2))
    return logp

def _generate_grid():
    z1 = torch.linspace(-4, 4)
    z2 = torch.linspace(-4, 4)
    grid_z1, grid_z2 = torch.meshgrid(z1, z2)
    grid = torch.stack([grid_z1, grid_z2], dim = -1)
    z = grid.reshape((-1, 2))
    return z
    
def test_logp():
    z = _generate_grid()
    
    fig = plt.figure(0)
    fig.clf()
    
    plt.subplot(2,2,1)
    logp1 = compute_logp1(z)
    logp1 = logp1.reshape(100, 100)    
    p1 = torch.exp(logp1)
    plt.imshow(p1.T, origin = "lower", extent = (-4,4,-4,4))
    plt.colorbar()
    
    plt.subplot(2,2,2)
    logp2 = compute_logp2(z)
    logp2 = logp2.reshape(100, 100)    
    p2 = torch.exp(logp2)
    plt.imshow(p2.T, origin = "lower", extent = (-4,4,-4,4))    
    plt.colorbar()

    plt.subplot(2,2,3)
    logp3 = compute_logp3(z)
    logp3 = logp3.reshape(100, 100)    
    p3 = torch.exp(logp3)
    plt.imshow(p3.T, origin = "lower", extent = (-4,4,-4,4))    
    plt.colorbar()
    
    plt.subplot(2,2,4)
    logp4 = compute_logp4(z)
    logp4 = logp4.reshape(100, 100)    
    p4 = torch.exp(logp4)
    plt.imshow(p4.T, origin = "lower", extent = (-4,4,-4,4))    
    plt.colorbar()
    
    plt.savefig("./output/true_prop.pdf")
    
if __name__ == "__main__":
    test_logp()
