import math

import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F

from models.stylegan2.model import EqualLinear, PixelNorm

class Mapper(Module):

    def __init__(self, in_channel=512, out_channel=512, norm=True, num_layers=4):
        super(Mapper, self).__init__()

        layers = [PixelNorm()] if norm else []
        
        layers.append(EqualLinear(in_channel, out_channel, lr_mul=0.01, activation='fused_lrelu'))
        for _ in range(num_layers-1):
            layers.append(EqualLinear(out_channel, out_channel, lr_mul=0.01, activation='fused_lrelu'))
        self.mapping = nn.Sequential(*layers)

    def forward(self, x):
        x = self.mapping(x)
        return x

class DeltaMapper(Module):

    def __init__(self):
        super(DeltaMapper, self).__init__()

        #Style Module(sm)
        self.sm_coarse = Mapper(512,  512)
        self.sm_medium = Mapper(512,  512)
        self.sm_fine   = Mapper(2464, 2464)

        #Condition Module(cm)
        self.cm_coarse = Mapper(1024, 512)
        self.cm_medium = Mapper(1024, 512)
        self.cm_fine   = Mapper(1024, 2464)

        #Fusion Module(fm)
        self.fm_coarse = Mapper(512*2,  512,  norm=False)
        self.fm_medium = Mapper(512*2,  512,  norm=False)
        self.fm_fine   = Mapper(2464*2, 2464, norm=False)
        
    def forward(self, sspace_feat, clip_feat):

        s_coarse = sspace_feat[:, :3*512].view(-1,3,512)
        s_medium = sspace_feat[:, 3*512:7*512].view(-1,4,512)
        s_fine   = sspace_feat[:, 7*512:] #channels:2464

        s_coarse = self.sm_coarse(s_coarse)
        s_medium = self.sm_medium(s_medium)
        s_fine   = self.sm_fine(s_fine)

        c_coarse = self.cm_coarse(clip_feat)
        c_medium = self.cm_medium(clip_feat)
        c_fine   = self.cm_fine(clip_feat)

        x_coarse = torch.cat([s_coarse, torch.stack([c_coarse]*3, dim=1)], dim=2) #[b,3,1024]
        x_medium = torch.cat([s_medium, torch.stack([c_medium]*4, dim=1)], dim=2) #[b,4,1024]
        x_fine   = torch.cat([s_fine, c_fine], dim=1) #[b,2464*2]

        x_coarse = self.fm_coarse(x_coarse)
        x_coarse = x_coarse.view(-1,3*512)

        x_medium = self.fm_medium(x_medium)
        x_medium = x_medium.view(-1,4*512)

        x_fine   = self.fm_fine(x_fine)

        out = torch.cat([x_coarse, x_medium, x_fine], dim=1)
        return out