import os
import sys
sys.path.append(".")
sys.path.append("..")

import copy
import clip
import numpy as np
from PIL import Image

import torch
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

import torch.nn.functional as F


from datasets.test_dataset import TestLatentsDataset

from models.stylegan2.model import Generator
from models.encoders import psp_encoders
from delta_mapper import DeltaMapper

from options.test_options import TestOptions

from utils import map_tool
from utils import stylespace_util



def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt

class Imagedataset(Dataset):
    def __init__(self,
                 path,
                 image_size=256,
                 split=None):

        self.path = path
        self.images = os.listdir(path)

        self.image_size = image_size

        self.length = len(self.images)

        transform = [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]

        self.transform = transforms.Compose(transform)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        cur_name = self.images[index]
        img_path = os.path.join(self.path, cur_name)

        img = Image.open(img_path).convert("RGB") 

        if self.transform is not None:
            img = self.transform(img)
        return img

def encoder_latent(G, latent):
    # an encoder warper for G
    #styles = [noise]
    style_space = []
    
    #styles = [G.style(s) for s in styles]
    noise = [getattr(G.noises, 'noise_{}'.format(i)) for i in range(G.num_layers)]
    # inject_index = G.n_latent
    #latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
    style_space.append(G.conv1.conv.modulation(latent[:, 0]))

    i = 1
    for conv1, conv2, to_rgb in zip(
        G.convs[::2], G.convs[1::2], G.to_rgbs
    ):
        style_space.append(conv1.conv.modulation(latent[:, i]))
        style_space.append(conv2.conv.modulation(latent[:, i+1]))
        i += 2
        
    return style_space, noise

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

    # NOTE load e4e
    checkpoint_path = "encoder4editing-main/e4e_ffhq_encode.pt"
    ckpt_enc = torch.load(checkpoint_path, map_location='cpu') #dict_keys(['state_dict', 'latent_avg', 'opts'])
    encoder = psp_encoders.Encoder4Editing(50, 1024, 'ir_se')
    encoder.load_state_dict(get_keys(ckpt_enc, 'encoder'), strict=True)
    encoder.eval()
    encoder.to(device)

    #Initialize test dataset
    test_dataset = Imagedataset('./test_imgs', image_size=256)
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
    avg_pool = torch.nn.AvgPool2d(kernel_size=256//32)
    upsample = torch.nn.Upsample(scale_factor=7)

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
        input_img = batch.to(device)
        with torch.no_grad():
            latent_w = encoder(input_img)
            latent_avg = ckpt_enc['latent_avg'].cuda()
            latent_w = latent_w + latent_avg.repeat(latent_w.shape[0], 1, 1)

            style_space, noise = encoder_latent(g_ema, latent_w)
            latent_s = torch.cat(style_space, dim=1)

            img_gen_for_clip = upsample(input_img)
            img_gen_for_clip = avg_pool(img_gen_for_clip)
            c_latents = clip_model.encode_image(img_gen_for_clip)
            c_latents = c_latents / c_latents.norm(dim=-1, keepdim=True).float()

        delta_s_list = []

        for i, dt in enumerate(dt_list):
            delta_c = torch.cat((c_latents, dt.unsqueeze(0)), dim=1)
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
