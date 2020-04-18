import torch 
import torch.nn as nn

def triplet_loss(x_a,x_p,x_n, margin = 1):
    d = nn.PairwiseDistance(p=2)
    distance = d(x_a,x_p).pow(2) - d(x_a,x_n).pow(2)
    return torch.max(distance+margin, torch.zeros_like(distance)).mean()
