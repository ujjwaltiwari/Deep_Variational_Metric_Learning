import argparse
import os
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader
import torch.distributions as dist
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import tqdm

from models.dvml import DVML
from datasets.base import make_infinite_iterator,BalancedBatchSampler,ZeroShotImageFolder
import utils
import losses

parser = argparse.ArgumentParser()

parser.add_argument("--z_dims", default = 512, type = int)
parser.add_argument("--log_dir", default = "logs", type = str)
parser.add_argument("--num_epochs", default = 100, type = int)
parser.add_argument("--samples_per_epoch", default = 60, type = int)
parser.add_argument("--learning_rate", default = 0.0001, type = float)
parser.add_argument("--data_root",default = "~/.data", type = str)
parser.add_argument("--n_classes", default = 6, type = int)
parser.add_argument("--n_samples", default = 20, type = int)
parser.add_argument("--checkpoint_freq",default = 2, type = int)
parser.add_argument("--eval_freq",default = 10, type = int)
parser.add_argument("--max_checkpoints", default = 5, type = int)
args = parser.parse_args()

#Load dataset
transform = transforms.Compose([transforms.RandomCrop((224,224),pad_if_needed=True),transforms.RandomHorizontalFlip(),transforms.ToTensor()])
train_dataset = ZeroShotImageFolder(args.data_root,train = True, transform = transform)
batch_sampler = BalancedBatchSampler(train_dataset,args.n_classes,args.n_samples)
train_loader = DataLoader(train_dataset,batch_sampler = batch_sampler, pin_memory=True)
train_loader = make_infinite_iterator(train_loader)


#Load model and optimizer
dvml_model = DVML(z_dims=args.z_dims).cuda()

optimizer = torch.optim.Adam(dvml_model.parameters(),lr = args.learning_rate)
start_itr = 0

#Load checkpoint if exist
checkpoint = utils.get_latest_checkpoint_file(args.log_dir)
if checkpoint:
    checkpoint = torch.load(checkpoint,torch.device(0))
    dvml_model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_itr = checkpoint["global_itr"]+1

#Load summary wirter
train_summary_writer = SummaryWriter(args.log_dir)

#Loss
margin_loss = losses.MarginLoss()
dws_sampler = losses.DistanceWeightedSampling(args.n_samples)
def triplet_dws_loss(x):
    _,x_a,x_p,x_n,_ = dws_sampler(x)
    return margin_loss(x_a,x_p,x_n)

pb = tqdm.tqdm(total = args.num_epochs,initial = start_itr)

for global_itr in range(start_itr,args.num_epochs):
    disc_sum = 0
    for i in range(args.samples_per_epoch):
        x,target = next(train_loader)
        x = x.cuda()
        target = target.cuda()
        z_inv,_,hidden = dvml_model(x)
        
        #Discriminate
        disc_loss = triplet_dws_loss(z_inv)

        optimizer.zero_grad()
        disc_loss.backward()
        optimizer.step()

        disc_sum += disc_inv_loss.item()
    pb.update()
    train_summary_writer.add_scalar("disc_loss",disc_sum/args.samples_per_epoch, global_step = global_itr)

    if global_itr%args.checkpoint_freq == 0:
        checkpoint = {
                    "model":dvml_model.state_dict(),
                    "optimizer":optimizer.state_dict(),
                    "global_itr":global_itr
                }
        utils.save_checkpoints(checkpoint,os.path.join(args.log_dir,"model_{}.pth".format(global_itr)),max_chkps = args.max_checkpoints)
    
checkpoint = {
    "model":dvml_model.state_dict(),
    "optimizer":optimizer.state_dict(),
    "global_itr":args.num_epochs-1
}
utils.save_checkpoints(checkpoint,os.path.join(args.log_dir,"model_{}.pth".format(args.num_epochs-1)),max_chkps = args.max_checkpoints)
train_summary_writer.close()
