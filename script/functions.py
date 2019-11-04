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

def _generate_grid():
    z1 = torch.linspace(-4, 4)
    z2 = torch.linspace(-4, 4)
    grid_z1, grid_z2 = torch.meshgrid(z1, z2)
    grid = torch.stack([grid_z1, grid_z2], dim = -1)
    z = grid.reshape((-1, 2))
    return z
    
def test_logp1():
    z = _generate_grid()
    logp1 = compute_logp1(z)
    logp1 = logp1.reshape(100, 100)
    p = torch.exp(logp1)
    fig = plt.figure(0)
    fig.clf()
    plt.imshow(p.T, origin = "lower", extent = (-4,4,-4,4))    
    plt.colorbar()
    plt.savefig("./output/true_p_1.pdf")
    
if __name__ == "__main__":
    test_logp1()
