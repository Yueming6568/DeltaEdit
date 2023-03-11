import os
import sys
sys.path.append(".")
sys.path.append("..")

import copy
import clip
import numpy as np

import torch
import torchvision
from torch.utils.data import DataLoader

import torch.nn.functional as F

from datasets.test_dataset import TestLatentsDataset

from models.stylegan2.model import Generator
from delta_mapper import DeltaMapper

from options.test_options import TestOptions

from utils import map_tool
from utils import stylespace_util

def GetBoundary(fs3,dt,threshold):
    tmp=np.dot(fs3,dt)
    
    select=np.abs(tmp)<threshold
    return select

def improved_ds(ds, select):
    ds_imp = copy.copy(ds)
    ds_imp[select] = 0
    ds_imp = ds_imp.unsqueeze(0)
    return ds_imp

def main(opts):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    #Initialize test dataset
    test_dataset = TestLatentsDataset()
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=opts.batch_size,
                                 shuffle=False,
                                 num_workers=int(opts.workers),
                                 drop_last=True)

    #Initialize generator
    print('Loading stylegan weights from pretrained!')
    g_ema = Generator(size=opts.stylegan_size, style_dim=512, n_mlp=8)
    g_ema_ckpt = torch.load(opts.stylegan_weights)
    g_ema.load_state_dict(g_ema_ckpt['g_ema'], strict=False)
    g_ema.eval()
    g_ema = g_ema.to(device)

    #load relevance matrix Rs
    fs3=np.load('./models/stylegan2/npy_ffhq/fs3.npy')
    np.set_printoptions(suppress=True)

    #Initialze DeltaMapper
    net = DeltaMapper()
    net_ckpt = torch.load(opts.checkpoint_path)
    net.load_state_dict(net_ckpt)
    net = net.to(device)
    
    #Load CLIP model
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    os.makedirs(opts.save_dir, exist_ok=True)

    neutral='face'
    target_list = opts.target.split(',')
    # print(target_list)

    dt_list = []
    select_list = []
    for target in target_list:
        classnames=[target,neutral]
        dt = map_tool.GetDt(classnames,clip_model)
        select = GetBoundary(fs3, dt, opts.threshold)
        dt = torch.Tensor(dt).to(device)
        dt = dt / dt.norm(dim=-1, keepdim=True).float().clamp(min=1e-5)

        select_list.append(select)
        dt_list.append(dt)

    for bid, batch in enumerate(test_dataloader):
        if bid == opts.num_all:
            break
        
        latent_s, delta_c, latent_w = batch
        latent_s = latent_s.to(device)
        delta_c = delta_c.to(device)
        latent_w = latent_w.to(device)
        delta_s_list = []

        for i, dt in enumerate(dt_list):
            delta_c[0, 512:] = dt
            with torch.no_grad():
                fake_delta_s = net(latent_s, delta_c)
                improved_fake_delta_s = improved_ds(fake_delta_s[0], select_list[i])
            delta_s_list.append(improved_fake_delta_s)

        with torch.no_grad():
            img_ori = stylespace_util.decoder_validate(g_ema, latent_s, latent_w)

            img_list = [img_ori]
            for delta_s in delta_s_list:
                img_gen = stylespace_util.decoder_validate(g_ema, latent_s + delta_s, latent_w)
                img_list.append(img_gen)
            img_gen_all = torch.cat(img_list, dim=3)
            torchvision.utils.save_image(img_gen_all, os.path.join(opts.save_dir, "%04d.jpg" %(bid+1)), normalize=True, range=(-1, 1))
    print(f'completed👍! Please check results in {opts.save_dir}')

if __name__ == "__main__":
    opts = TestOptions().parse()
    main(opts)