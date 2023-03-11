import copy
import random
import numpy as np

import torch
from torch.utils.data import Dataset

class TrainLatentsDataset(Dataset):
    def __init__(self, opts, cycle=True):

        style_latents_list = []
        clip_latents_list = []
        wplus_latents_list = []

        style_latents_list.append(torch.Tensor(np.load(f"./latent_code/{opts.classname}/sspace_noise_feat.npy")))
        clip_latents_list.append(torch.Tensor(np.load(f"./latent_code/{opts.classname}/cspace_noise_feat.npy")))
        wplus_latents_list.append(torch.Tensor(np.load(f"./latent_code/{opts.classname}/wspace_noise_feat.npy")))
        
        style_latents_list.append(torch.Tensor(np.load(f"./latent_code/{opts.classname}/sspace_ffhq_feat.npy")))
        clip_latents_list.append(torch.Tensor(np.load(f"./latent_code/{opts.classname}/cspace_ffhq_feat.npy")))
        wplus_latents_list.append(torch.Tensor(np.load(f"./latent_code/{opts.classname}/wspace_ffhq_feat.npy")))
        
        self.style_latents = torch.cat(style_latents_list, dim=0)
        self.clip_latents = torch.cat(clip_latents_list, dim=0)
        self.wplus_latents = torch.cat(wplus_latents_list, dim=0)

        self.style_latents = self.style_latents[:200000+58000]
        self.clip_latents = self.clip_latents[:200000+58000]
        self.wplus_latents = self.wplus_latents[:200000+58000]

        self.dataset_size = self.style_latents.shape[0]
        print("dataset size", self.dataset_size)
        self.cycle = cycle
        
    def __len__(self):
        if self.cycle:
            return self.style_latents.shape[0] * 50
        else:
            return self.style_latents.shape[0]

    def __getitem__(self, index):
        if self.cycle:
            index = index % self.dataset_size

        latent_s1 = self.style_latents[index]
        latent_c1 = self.clip_latents[index]
        latent_w1 = self.wplus_latents[index]
        latent_c1 = latent_c1 / latent_c1.norm(dim=-1, keepdim=True).float()

        random_index = random.randint(0, self.dataset_size - 1)
        latent_s2 = self.style_latents[random_index]
        latent_c2 = self.clip_latents[random_index]
        latent_w2 = self.wplus_latents[random_index]
        latent_c2 = latent_c2 / latent_c2.norm(dim=-1, keepdim=True).float()

        delta_s1 = latent_s2 - latent_s1
        delta_c = latent_c2 - latent_c1
        
        delta_c = delta_c / delta_c.norm(dim=-1, keepdim=True).float().clamp(min=1e-5)
        delta_c = torch.cat([latent_c1, delta_c], dim=0)

        return latent_s1, delta_c, delta_s1