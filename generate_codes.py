import os
import argparse
import clip

import random
import numpy as np
import torch
from torchvision import utils
from utils import stylespace_util
from models.stylegan2.model import Generator

def save_image_pytorch(img, name):
    """Helper function to save torch tensor into an image file."""
    utils.save_image(
        img,
        name,
        nrow=1,
        padding=0,
        normalize=True,
        range=(-1, 1),
    )


def generate(args, netG, device, mean_latent):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    avg_pool = torch.nn.AvgPool2d(kernel_size=1024 // 32)
    upsample = torch.nn.Upsample(scale_factor=7)

    ind = 0
    with torch.no_grad():
        netG.eval()

        # Generate images from a file of input noises
        if args.fixed_z is not None:
            sample_z = torch.load(args.fixed_z, map_location=device)
            for start in range(0, sample_z.size(0), args.batch_size):
                end = min(start + args.batch_size, sample_z.size(0))
                z_batch = sample_z[start:end]
                sample, _ = netG([z_batch], truncation=args.truncation, truncation_latent=mean_latent)
                for s in sample:
                    save_image_pytorch(s, f'{args.save_dir}/{str(ind).zfill(6)}.png')
                    ind += 1
            return

        # Generate image by sampling input noises
        w_latents_list = []
        s_latents_list = []
        c_latents_list = []
        for start in range(0, args.samples, args.batch_size):
            end = min(start + args.batch_size, args.samples)
            batch_sz = end - start
            print(f'current_num:{start}')
            sample_z = torch.randn(batch_sz, 512, device=device)

            sample, w_latents = netG([sample_z], truncation=args.truncation, truncation_latent=mean_latent,return_latents=True)
            style_space, noise = stylespace_util.encoder_latent(netG, w_latents)
            s_latents = torch.cat(style_space, dim=1)

            tmp_imgs = stylespace_util.decoder(netG, style_space, w_latents, noise)
            # for s in tmp_imgs:
            #     save_image_pytorch(s, f'{args.save_dir}/{str(ind).zfill(6)}.png')
            #     ind += 1

            img_gen_for_clip = upsample(tmp_imgs)
            img_gen_for_clip = avg_pool(img_gen_for_clip)
            c_latents = model.encode_image(img_gen_for_clip)

            w_latents_list.append(w_latents)
            s_latents_list.append(s_latents)
            c_latents_list.append(c_latents)
        w_all_latents = torch.cat(w_latents_list, dim=0)
        s_all_latents = torch.cat(s_latents_list, dim=0)
        c_all_latents = torch.cat(c_latents_list, dim=0)

        print(w_all_latents.size())
        print(s_all_latents.size())
        print(c_all_latents.size())

        w_all_latents = w_all_latents.cpu().numpy()
        s_all_latents = s_all_latents.cpu().numpy()
        c_all_latents = c_all_latents.cpu().numpy()

        os.makedirs(os.path.join(args.save_dir, args.classname), exist_ok=True)
        np.save(f"{args.save_dir}/{args.classname}/wspace_noise_feat.npy", w_all_latents)
        np.save(f"{args.save_dir}/{args.classname}/sspace_noise_feat.npy", s_all_latents)
        np.save(f"{args.save_dir}/{args.classname}/cspace_noise_feat.npy", c_all_latents)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--classname', type=str, default='ffhq', help="place to save the output")
    parser.add_argument('--save_dir', type=str, default='./latent_code', help="place to save the output")
    parser.add_argument('--ckpt', type=str, default='./models/pretrained_models', help="checkpoint file for the generator")
    parser.add_argument('--size', type=int, default=1024, help="output size of the generator")
    parser.add_argument('--fixed_z', type=str, default=None, help="expect a .pth file. If given, will use this file as the input noise for the output")
    parser.add_argument('--w_shift', type=str, default=None, help="expect a .pth file. Apply a w-latent shift to the generator")
    parser.add_argument('--batch_size', type=int, default=10, help="batch size used to generate outputs")
    parser.add_argument('--samples', type=int, default=200000, help="200000 number of samples to generate, will be overridden if --fixed_z is given")
    parser.add_argument('--truncation', type=float, default=1, help="strength of truncation:0.5ori")
    parser.add_argument('--truncation_mean', type=int, default=4096, help="number of samples to calculate the mean latent for truncation")
    parser.add_argument('--seed', type=int, default=None, help="if specified, use a fixed random seed")
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    device = args.device
    # use a fixed seed if given
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    netG = Generator(args.size, 512, 8).to(device)
    if args.classname == 'ffhq':
        ckpt_path = os.path.join(args.ckpt,f'stylegan2-{args.classname}-config-f.pt')
    else:
        ckpt_path = os.path.join(args.ckpt,f'stylegan2-{args.classname}','netG.pth')
    print(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    if args.classname == 'ffhq':
        netG.load_state_dict(checkpoint['g_ema'])
    else:
        netG.load_state_dict(checkpoint)

    # get mean latent if truncation is applied
    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = netG.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    generate(args, netG, device, mean_latent)
