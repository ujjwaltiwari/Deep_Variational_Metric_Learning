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
parser.add_argument("--num_first_phase_epochs", default = 25, type = int)
args = parser.parse_args()

#Load dataset
transform = transforms.Compose([transforms.RandomCrop((224,224),pad_if_needed=True),transforms.RandomHorizontalFlip(),transforms.ToTensor()])
train_dataset = ZeroShotImageFolder(args.data_root,train = True, transform = transform)
batch_sampler = BalancedBatchSampler(train_dataset,args.n_classes,args.n_samples)
train_loader = DataLoader(train_dataset,batch_sampler = batch_sampler, pin_memory=True)
train_loader = make_infinite_iterator(train_loader)


#Load model and optimizer
dvml_model = DVML(z_dims=args.z_dims).cuda()
decoder_model = nn.Sequential(
            nn.Linear(args.z_dims,512),
            nn.Tanh(),
            nn.Linear(512,1024)
        ).cuda()

optimizer = torch.optim.Adam(itertools.chain(dvml_model.parameters(),decoder_model.parameters()),lr = args.learning_rate)
start_itr = 0

#Load checkpoint if exist
checkpoint = utils.get_latest_checkpoint_file(args.log_dir)
if checkpoint:
    checkpoint = torch.load(checkpoint,torch.device(0))
    dvml_model.load_state_dict(checkpoint["dvml"])
    decoder_model.load_state_dict(checkpoint["decoder"])
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
    phase1 = global_itr < args.num_first_phase_epochs
    l2_sum = 0
    kl_sum = 0
    disc_inv_sum = 0
    disc_latent_sum = 0
    total_sum = 0
    for i in range(args.samples_per_epoch):
        x,target = next(train_loader)
        x = x.cuda()
        target = target.cuda()
        z_inv,mu_logvar,hidden = dvml_model(x)

        q_dist = utils.get_normal_from_params(mu_logvar)
        p_dist = dist.Normal(0,1)

        #KL Loss
        kl_loss = dist.kl_divergence(q_dist,p_dist).sum(-1).mean()

        #Discriminate invar
        disc_inv_loss = triplet_dws_loss(z_inv)

        if phase1:
            z_var = q_dist.sample()
            z = z_inv + z_var
            decoded = decoder_model(z.detach())
            #l2_loss
            l2_loss = F.mse_loss(hidden,decoded)
        else:
            z_var = q_dist.sample((20,))
            z = z_inv[None] + z_var
            decoded = decoder_model(z.reshape([-1,args.z_dims]))
            decoded = decoded.reshape([20,-1,1024])
            #l2_loss
            l2_loss = F.mse_loss(hidden,decoded)
            z = z[0]


        #discriminate latent
        disc_latent_loss = triplet_dws_loss(z)

        if phase1:
            total_loss = kl_loss + l2_loss + 0.1 * disc_latent_loss + disc_inv_loss
        else:
            total_loss = 0.8*kl_loss + l2_loss + 0.2 * disc_latent_loss + 0.8*disc_inv_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        total_sum += total_loss.item()
        kl_sum += kl_loss.item()
        l2_sum += l2_loss.item()
        disc_inv_sum += disc_inv_loss.item()
        disc_latent_sum += disc_latent_loss.item()
    pb.update()
    train_summary_writer.add_scalar("total_loss",total_sum/args.samples_per_epoch,global_step = global_itr)
    train_summary_writer.add_scalar("kl_loss",kl_loss/args.samples_per_epoch, global_step = global_itr)
    train_summary_writer.add_scalar("l2_loss",l2_sum/args.samples_per_epoch, global_step = global_itr)
    train_summary_writer.add_scalar("disc_inv_loss",disc_inv_sum/args.samples_per_epoch, global_step = global_itr)
    train_summary_writer.add_scalar("disc_latent_loss",disc_latent_sum/args.samples_per_epoch, global_step = global_itr)

    if global_itr%args.checkpoint_freq == 0:
        checkpoint = {
                    "dvml":dvml_model.state_dict(),
                    "decoder":decoder_model.state_dict(),
                    "optimizer":optimizer.state_dict(),
                    "global_itr":global_itr
                }
        utils.save_checkpoints(checkpoint,os.path.join(args.log_dir,"model_{}.pth".format(global_itr)),max_chkps = args.max_checkpoints)
    
checkpoint = {
    "dvml":dvml_model.state_dict(),
    "decoder":decoder_model.state_dict(),
    "optimizer":optimizer.state_dict(),
    "global_itr":args.num_epochs-1
}
utils.save_checkpoints(checkpoint,os.path.join(args.log_dir,"model_{}.pth".format(args.num_epochs-1)),max_chkps = args.max_checkpoints)
train_summary_writer.close()
