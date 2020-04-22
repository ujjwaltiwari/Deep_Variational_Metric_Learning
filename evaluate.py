import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.neighbors import NearestNeighbors

from models.dvml import DVML
from datasets.base import ZeroShotImageFolder
import utils


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint",type = str)
parser.add_argument("--z_dims", default = 512, type = int)
parser.add_argument("--data_root",type = str)
args = parser.parse_args()

model = DVML(z_dims = args.z_dims).cuda()
state_dict = torch.load(args.checkpoint,map_location = torch.device(0))
model.load_state_dict(state_dict["dvml"])
model.eval()

transform = transforms.Compose([transforms.CenterCrop((224,224)),transforms.ToTensor()])
dataset = ZeroShotImageFolder(args.data_root,train=False,transform = transform)
dataloader = DataLoader(dataset, batch_size = 50)
z_all = []
targets_all = []

for images,target in dataloader:
    if torch.no_grad():
        import ipdb; ipdb.set_trace()
        z_inv,mu_logvar,_ = model(images.cuda())
        z_var = utils.sample_normal(mu_logvar)
        z = z_inv + z_var
    z_all.append(z.detach())
    targets_all.append(target)

z_all = torch.cat(z_all,0).cpu().numpy()
targets_all = torch.cat(targets_all,0).numpy()

scores = []
for n in [1,2,4,8]:
    nbrs = NearestNeighbors(n_neighbors=n).fit(z_all)
    indices = nbrs.kneighbors(z_all,n_neighbors=n+1,return_distance=False)
    nbr_targets = targets_all[indices[:,1:]]
    nbr_targets = np.any(nbr_targets == targets_all[:,np.newaxis],axis =-1)
    score = sum(nbr_targets)/nbr_targets.shape[0]*100
    print("n = ",n,", score = ",score)
    scores.append(score)

print(scores)
