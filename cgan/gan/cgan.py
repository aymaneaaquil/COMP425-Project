import argparse
import os
import numpy as np
import math
import datetime
import json
import glob

import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from datasets import *
from models import *

import torch.nn as nn
import torch.nn.functional as F
import torch

#most of it with some exceptions here and there are from https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/ccgan

# So this is a function I needed to add, it was not present initially, I pretty much just initialize the weights of the generator and discrimnator nn. This gets applied to every layer of each NN
# so pretty much, this is a little optimization step, where we overwrite the defaults a bit to something that will work better
# how did I know what to set it to? this article https://arxiv.org/abs/1511.06434, mentioned that all starting values where 0 centered with a std dev of 0.02. thats for Conv. and as for BatchNorms2d, again in that last article we saw std of 0.02 but I read here that someone had done an init with 1.0, https://discuss.pytorch.org/t/batchnorm-initialization/16184, so I just tried that. I also decided that we cna set the bias weight to 0. didnt see a reason to set it to something else.
# this optimization if done well can improve convergence.
def weights_init_normal(m):
    classname = m.__class__.__name__
    #if the classname has Conv somewhere in it, with a mean of 0 and a std dev of 0.02.
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        # here we make it 1 centered, and also set all the bias params to 0.
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# Here we diverge a bit from what was done in the normal code, I want to put all the code for every model, in its own folder, so it will be easier to go through myself later. I mean really this is just a little cleanup move.
os.makedirs("gan/models", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--dataset_name", type=str, default="data/processed_data/train", help="name of the dataset")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--mask_size", type=int, default=32, help="size of random mask")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=50, help="interval between image sampling")
# I added another param here called mode_save_interval, this is how often we overwrite the model.pth, pretty much these files are big, and I dont want to just save it every epoch or something, because I think it is bad for my ssd, because it is way to many writes.
parser.add_argument("--model_save_interval", type=int, default=1, help="interval (in epochs) to save model checkpoints")
opt = parser.parse_args()
print(opt)


# Here we actually make that dir, so early we make the models dir if needed, and here we made a new sub dir, one for  every model. I just save as date, with a split after day. just my convention, I am sure there are others but this one gets the job done.
timestamp = datetime.datetime.now().strftime("%d%m%y-%H%M%S")
model_dir = os.path.join("gan/models", timestamp)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(os.path.join(model_dir, "images"), exist_ok=True)

# Right when we start training that model, in that subdir, we save in config.json the param we used to train it.
with open(os.path.join(model_dir, "config.json"), "w") as f:
    json.dump(vars(opt), f, indent=4)

# if we have nvidia gpu, we can use cude, otherwise we dont.
cuda = True if torch.cuda.is_available() else False

# here we find the input_shape of the image, from the params above
input_shape = (opt.channels, opt.img_size, opt.img_size)

# Loss function
adversarial_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = Generator(input_shape)
discriminator = Discriminator(input_shape)

#again we check if we have Cuda
if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights (we talked about this earlier)
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Dataset loader, these are the transforms we will be using when we train our model, every time we want 2 images, a low res and a high res, this is how we will be asking for them
transforms_ = [
    transforms.Resize((opt.img_size, opt.img_size), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
transforms_lr = [
    transforms.Resize((opt.img_size // 4, opt.img_size // 4), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
#load the dataloader
dataloader = DataLoader(
    ImageDataset(opt.dataset_name, transforms_x=transforms_, transforms_lr=transforms_lr),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

# Optimizers, we are using Adam, I have had great experience and I think it is A tier.
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

#this is where we apply that black box to the image, the part that we will try to fill in.
def apply_random_mask(imgs):
    idx = np.random.randint(0, opt.img_size - opt.mask_size, (imgs.shape[0], 2))

    masked_imgs = imgs.clone()
    for i, (y1, x1) in enumerate(idx):
        y2, x2 = y1 + opt.mask_size, x1 + opt.mask_size
        masked_imgs[i, :, y1:y2, x1:x2] = -1

    return masked_imgs


def save_sample(saved_samples, batches_done):
    # Generate inpainted image
    gen_imgs = generator(saved_samples["masked"], saved_samples["lowres"])
    # Save sample
    sample = torch.cat((saved_samples["masked"].data, gen_imgs.data, saved_samples["imgs"].data), -2)
    
    # made a small change here, we save only to a model specific folder.
    save_image(sample, f"{model_dir}/images/{batches_done}.png", nrow=5, normalize=True)


#made a little custom function for the cleanup, I dont want to blow up my ssd with lots of models, because I am doing alot of trainings, I will just delete all models on each save except the last 2.
def cleanup_model_files(model_dir):
    #get all pth files, if there are more than 2, sort and remove all but the 2 newest
    gen_files = glob.glob(os.path.join(model_dir, "generator_*.pth"))
    if len(gen_files) > 2:
        gen_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        for file in gen_files[:-2]:
            os.remove(file)
    #same idea for discriminator, not sure if i will need but probably not bad to have.
    disc_files = glob.glob(os.path.join(model_dir, "discriminator_*.pth"))
    if len(disc_files) > 2:
        disc_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        for file in disc_files[:-2]:
            os.remove(file)


# he we just init some stats we will be keeping track of.
stats = {"epoch": [], "d_loss": [], "g_loss": []}

# this is our main training script
#made some small changes here, we now track the losses.
saved_samples = {}
for epoch in range(opt.n_epochs):
    epoch_d_loss = 0
    epoch_g_loss = 0
    batches_in_epoch = 0
    
    for i, batch in enumerate(dataloader):
        imgs = batch["x"]
        imgs_lr = batch["x_lr"]

        masked_imgs = apply_random_mask(imgs)

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], *discriminator.output_shape).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], *discriminator.output_shape).fill_(0.0), requires_grad=False)

        if cuda:
            imgs = imgs.type(Tensor)
            imgs_lr = imgs_lr.type(Tensor)
            masked_imgs = masked_imgs.type(Tensor)

        real_imgs = Variable(imgs)
        imgs_lr = Variable(imgs_lr)
        masked_imgs = Variable(masked_imgs)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Generate a batch of images
        gen_imgs = generator(masked_imgs, imgs_lr)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()

        epoch_d_loss += d_loss.item()
        epoch_g_loss += g_loss.item()
        batches_in_epoch += 1

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        # Save first ten samples
        if not saved_samples:
            saved_samples["imgs"] = real_imgs[:1].clone()
            saved_samples["masked"] = masked_imgs[:1].clone()
            saved_samples["lowres"] = imgs_lr[:1].clone()
        elif saved_samples["imgs"].size(0) < 10:
            saved_samples["imgs"] = torch.cat((saved_samples["imgs"], real_imgs[:1]), 0)
            saved_samples["masked"] = torch.cat((saved_samples["masked"], masked_imgs[:1]), 0)
            saved_samples["lowres"] = torch.cat((saved_samples["lowres"], imgs_lr[:1]), 0)

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_sample(saved_samples, batches_done)
    
    # ok now here are things that I added
    # here we decide if its time to save, and if it is, we save
    if (epoch + 1) % opt.model_save_interval == 0 or epoch == opt.n_epochs - 1:
        # Save the current generator and discriminator models
        torch.save(generator.state_dict(), os.path.join(model_dir, f"generator_{epoch+1}.pth"))
        torch.save(discriminator.state_dict(), os.path.join(model_dir, f"discriminator_{epoch+1}.pth"))
        
        # now we do our little cleanup
        cleanup_model_files(model_dir)
        
        print(f"Models saved at epoch {epoch+1} and old models cleaned up")
    
    # now here we measure some stats and save them. why do we check for batches > 0 ? this will get triggered most of the time but if it is 0, then by checking we prevent divisions by 0 later
    if batches_in_epoch > 0:
        avg_d_loss = epoch_d_loss / batches_in_epoch
        avg_g_loss = epoch_g_loss / batches_in_epoch
        
        stats["epoch"].append(epoch + 1)
        stats["d_loss"].append(avg_d_loss)
        stats["g_loss"].append(avg_g_loss)
        
        # Update training stats file
        with open(os.path.join(model_dir, "training_stats.json"), "w") as f:
            json.dump(stats, f, indent=4)