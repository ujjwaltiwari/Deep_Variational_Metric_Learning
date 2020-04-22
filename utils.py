import glob
import re
import os

import torch
import torch.nn as nn
import torch.distributions as dist
from bottleneck import argpartition


def get_normal_from_params(params, dim = 1):
    mu, log_var = torch.chunk(params,2,dim=dim)
    std = torch.exp(log_var*0.5)
    distribution = dist.Normal(mu,std)
    return distribution

def sample_normal(params,dim = 1):
    normal = get_normal_from_params(params, dim = dim)
    sample = normal.rsample()
    return sample

def save_checkpoints(checkpoint, name, max_chkps = 5):
    #Find all checkpoints in the folder
    dirname = os.path.dirname(name)
    checkpoint_files = glob.glob(os.path.join(dirname,"*.pth"))
    if len(checkpoint_files) >= max_chkps:
        checkpoint_files.sort(key = lambda x: int(re.findall(r"\d+",x)[-1]))
        os.remove(checkpoint_files[0])
    torch.save(checkpoint, name)

def get_latest_checkpoint_file(dir):
    checkpoint_files = glob.glob(os.path.join(dir,"*.pth"))
    checkpoint_files.sort(key = lambda x: int(re.findall(r"\d+",x)[-1]),reverse = True)
    if checkpoint_files:
        return checkpoint_files[0]
    else:
        return None

