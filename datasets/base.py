import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler
from torchvision.datasets import ImageFolder

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


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples):
        loader = DataLoader(dataset)
        self.labels_list = []
        for _, label in loader:
            self.labels_list.append(label)
        self.labels = torch.LongTensor(self.labels_list)
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size

