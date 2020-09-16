#!/usr/bin/env python
#coding:utf8
import os
import numpy as np
import random
from torchvision import  transforms as T
from torch.utils import data
import torch

###
class dataread(data.Dataset):
    def __init__(self, root, transforms = None, train = True, test = False, val = False):
        self.root = root
        self.train = train
        self.val = val
        self.test = test

        if self.train:
            self.folderlist = os.listdir(os.path.join(self.root))
        elif self.val:
            self.folderlist = os.listdir(os.path.join(self.root))
        elif self.test:
            self.folderlist = os.listdir(os.path.join(self.root))

    def __getitem__(self,index):
        
        img = np.load(os.path.join(self.root,self.folderlist[index]))
        img = np.asarray(img)
        
        if self.train:
            transform=transforms.Compose([transforms.RandomSizedCrop((128,128,128)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(),
                                         ])
        if self.val:
            transform=transforms.Compose([transforms.CenterCrop((128,128,128)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(),
                                         ])
        
        img_out = transform(img)[0:4,:,:,:]
        label_out = transform(img)[4,:,:,:]     
            
        return img_out, label_out

    def __len__(self):
        return len(self.folderlist)







