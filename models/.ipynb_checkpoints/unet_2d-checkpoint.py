# 3D-UNet model.
import torch
import torch.nn as nn
from utils import *

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()  

def conv_block_2d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        activation,)

def conv_trans_block_2d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(out_dim),
        activation,)

def max_pooling_2d():
    return nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

def conv_block_2_2d(in_dim, out_dim, activation):
    return nn.Sequential(
        conv_block_2d(in_dim, out_dim, activation),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),)

def conv_block_2_2d_layer1(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        activation,
        nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim))

def conv_block_2_2d_layer2(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        activation,
        nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim))

class unet_2d(nn.Module):
    def __init__(self, in_dim=4, classes=5):
        super(unet_2d, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = classes
        self.filters_list = [8, 16, 32, 64]
        activation = nn.LeakyReLU(0.2, inplace=True)
        
        # Down sampling
        self.down_1 = conv_block_2_2d_layer1(self.in_dim, self.filters_list[0], activation)
        self.pool_1 = max_pooling_2d()
        self.down_2 = conv_block_2_2d_layer2(self.filters_list[0], self.filters_list[1], activation)
        self.pool_2 = max_pooling_2d()
        self.down_3 = conv_block_2_2d_layer2(self.filters_list[1], self.filters_list[2], activation)
        self.pool_3 = max_pooling_2d()
        
        # Bridge
        self.bridge = conv_block_2_2d(self.filters_list[2], self.filters_list[3], activation)
        
        # Up sampling
        self.trans_1 = conv_trans_block_2d(self.filters_list[3], self.filters_list[3], activation)
        self.up_1 = conv_block_2_2d(self.filters_list[3]+self.filters_list[2], self.filters_list[2], activation)
        self.trans_2 = conv_trans_block_2d(self.filters_list[2], self.filters_list[2], activation)
        self.up_2 = conv_block_2_2d(self.filters_list[2]+self.filters_list[1], self.filters_list[1], activation)
        self.trans_3 = conv_trans_block_2d(self.filters_list[1], self.filters_list[1], activation)
        self.up_3 = conv_block_2_2d(self.filters_list[1]+self.filters_list[0], self.filters_list[0], activation)
        
        # Output
        self.out = conv_block_2d(self.filters_list[0], classes, activation)

        #self.final_activation = nn.Softmax(dim=1)

        initialize_weights(self)
    
    def forward(self, x):
        # Down sampling
        down_1 = self.down_1(x) 
        pool_1 = self.pool_1(down_1)
        
        down_2 = self.down_2(pool_1) 
        pool_2 = self.pool_2(down_2)
        
        down_3 = self.down_3(pool_2) 
        pool_3 = self.pool_3(down_3) 
        
        # Bridge
        bridge = self.bridge(pool_3)

        # Up sampling
        trans_1 = self.trans_1(bridge) 
        concat_1 = torch.cat([trans_1, down_3], dim=1) 
        up_1 = self.up_1(concat_1)
        
        trans_2 = self.trans_2(up_1) 
        concat_2 = torch.cat([trans_2, down_2], dim=1) 
        up_2 = self.up_2(concat_2) 
        
        trans_3 = self.trans_3(up_2)
        concat_3 = torch.cat([trans_3, down_1], dim=1) 
        up_3 = self.up_3(concat_3) 
        
        # Output
        out = self.out(up_3)

        #out = self.final_activation(out)

        return out

