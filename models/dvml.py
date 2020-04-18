import torch
import torch.nn as nn

from models.googletnet import googlenet
import utils

class DVML(nn.Module):
    def __init__(self,z_dims = 512):
        super().__init__()
        self.z_dims = z_dims
        
        self.googletnet = googlenet()
        self.fc_mu_logvar = nn.Linear(1024, 2*z_dims)
        self.fc_inv = nn.Linear(1024,z_dims)

    def forward(self,x):
        hidden = self.googletnet(x)
        mu_logvar = self.fc_mu_logvar(hidden)
        z_inv = self.fc_inv(hidden)
        return z_inv,mu_logvar,hidden
