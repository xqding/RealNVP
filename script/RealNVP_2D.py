__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2019/11/03 20:46:23"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as functional

class Affine_Coupling(nn.Module):
    def __init__(self, mask, hidden_dim):
        super(Affine_Coupling, self).__init__()
        self.input_dim = len(mask)
        self.hidden_dim = hidden_dim
        
        self.mask = nn.Parameter(mask, requires_grad = False)
        
        self.scale_fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.scale_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.scale_fc3 = nn.Linear(self.hidden_dim, self.input_dim)
        self.scale = nn.Parameter(torch.Tensor(self.input_dim))
        init.normal_(self.scale)
        
        self.translation_fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.translation_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.translation_fc3 = nn.Linear(self.hidden_dim, self.input_dim)
        
    def forward(self, x):
        s = functional.leaky_relu(self.scale_fc1(x*self.mask))
        s = functional.leaky_relu(self.scale_fc2(s))
        s = torch.tanh(self.scale_fc3(s)) * self.scale
        
        t = functional.leaky_relu(self.translation_fc1(x*self.mask))
        t = functional.leaky_relu(self.translation_fc2(t))
        t = self.translation_fc3(t)
        
        y = self.mask*x + (1-self.mask)*(x*torch.exp(s) + t)
        
        logdet = torch.sum((1 - self.mask)*s, -1)
        
        return y, logdet

    def inverse(self, y):
        s = torch.relu(self.scale_fc1(y*self.mask))
        s = torch.relu(self.scale_fc2(s))
        s = torch.tanh(self.scale_fc3(s)) * self.scale

        t = torch.relu(self.translation_fc1(y*self.mask))
        t = torch.relu(self.translation_fc2(t))
        t = self.translation_fc3(t)
        
        x = self.mask*y + (1-self.mask)*((y - t)*torch.exp(-s))
        logdet = torch.sum((1 - self.mask)*(-s), -1)
        
        return x, logdet

    
class RealNVP_2D(nn.Module):    
    def __init__(self, masks, hidden_dim):
        super(RealNVP_2D, self).__init__()
        self.hidden_dim = hidden_dim        
        self.masks = nn.ParameterList(
            [nn.Parameter(torch.Tensor(m),requires_grad = False)
             for m in masks])

        self.affine_couplings = nn.ModuleList(
            [Affine_Coupling(self.masks[i], self.hidden_dim)
             for i in range(len(self.masks))])
        
    def forward(self, x):
        y = x
        logdet_tot = 0
        for i in range(len(self.affine_couplings)):
            y, logdet = self.affine_couplings[i](y)
            logdet_tot = logdet_tot + logdet
                
        return y, logdet_tot

    def inverse(self, y):
        x = y
        logdet_tot = 0
        for i in range(len(self.affine_couplings)-1, -1, -1):
            x, logdet = self.affine_couplings[i].inverse(x)
            logdet_tot = logdet_tot + logdet
            
        return x, logdet_tot
