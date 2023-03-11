import numpy as np

import torch
from torch.utils.data import Dataset

class TestLatentsDataset(Dataset):
    def __init__(self):

        style_latents_list = []
        clip_latents_list = []
        wplus_latents_list = []
        
        #change the paths here for testing other latent codes
        style_latents_list.append(torch.Tensor(np.load("./examples/sspace_img_feat.npy")))
        clip_latents_list.append(torch.Tensor(np.load("./examples/cspace_img_feat.npy")))
        wplus_latents_list.append(torch.Tensor(np.load("./examples/wplus_img_feat.npy")))
        
        self.style_latents = torch.cat(style_latents_list, dim=0)
        self.clip_latents = torch.cat(clip_latents_list, dim=0)
        self.wplus_latents = torch.cat(wplus_latents_list, dim=0)
        
    def __len__(self):

        return self.style_latents.shape[0]

    def __getitem__(self, index):

        latent_s1 = self.style_latents[index]
        latent_c1 = self.clip_latents[index]
        latent_w1 = self.wplus_latents[index]
        latent_c1 = latent_c1 / latent_c1.norm(dim=-1, keepdim=True).float()
        
        delta_c = torch.cat([latent_c1, latent_c1], dim=0)
        
        return latent_s1, delta_c, latent_w1