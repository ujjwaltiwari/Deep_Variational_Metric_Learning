import random

import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
import numpy as np

def make_infinite_iterator(data_loader):
    data_itr = iter(data_loader)
    while True:
        try:
            x = next(data_itr)
        except StopIteration:
            data_itr = iter(data_loader)
            x = next(data_itr)
        yield x

class ZeroShotImageFolder(ImageFolder):
    def __init__(self,root,train=True,**kwargs):
        super().__init__(root,**kwargs)
        self.train = train
        self.target_set = list(set(self.targets))
        if train:
            self.target_set = self.target_set[:len(self.target_set)//2]
        else:
            self.target_set = self.target_set[len(self.target_set)//2:] 
        self.target_set = set(self.target_set)
        
        self.indices = [(idx,target) for idx,target in enumerate(self.targets) if target in self.target_set]
        self.indices, self.targets = zip(*self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self,index):
        return super().__getitem__(self.indices[index])

class TripletImageFolder(ZeroShotImageFolder):
    def __init__(self,root,train=True,**kwargs):
        super().__init__(root,train=train,**kwargs)
        self.train = train

        self.target_to_idx = {
                    target:[idx for idx,temp_target in enumerate(self.targets) if temp_target == target] for target in list(self.target_set)
                }

    def __getitem__(self,index):
        image, label = super().__getitem__(index)
        positive_idx = random.choice(self.target_to_idx[label])
        negative_idx = random.choice(self.target_to_idx[random.choice(list(self.target_set - set([label])))])
        pos_image,_ = super().__getitem__(positive_idx)
        neg_image,_ = super().__getitem__(negative_idx)
        return image, pos_image, neg_image
