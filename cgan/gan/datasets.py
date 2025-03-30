import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
#most of it with some exceptions here and there are from https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/ccgan

#This is a helper file that loads images from a target dataset (data/processed_data/train)

#during initialization, we find all the images in that folder. then every time we get asked for a photo, we make a high res "x" and a low res "x_lr", and then return them as a dict. we make each of the transforms using the transforms param, and we apply it here.

#TLDR it gets called saying give me 2 images (of the same image), with the following transforms, and then it does it.

class ImageDataset(Dataset):
    def __init__(self, root, transforms_x=None, transforms_lr=None, mode='train'):
        self.transform_x = transforms.Compose(transforms_x)
        self.transform_lr = transforms.Compose(transforms_lr)

        self.files = sorted(glob.glob('%s/*.*' % root))

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])

        x = self.transform_x(img)
        x_lr = self.transform_lr(img)

        return {'x': x, 'x_lr': x_lr}

    def __len__(self):
        return len(self.files)