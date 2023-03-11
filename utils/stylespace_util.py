import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision

from torch.nn import functional as F

index = [0,1,1,2,2,3,4,4,5,6,6,7,8,8,9,10,10,11,12,12,13,14,14,15,16,16]

def conv_warper(layer, input, style, noise):

    conv = layer.conv
    batch, in_channel, height, width = input.shape

    style = style.view(batch, 1, in_channel, 1, 1)
    weight = conv.scale * conv.weight * style

    if conv.demodulate:
        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
        weight = weight * demod.view(batch, conv.out_channel, 1, 1, 1)

    weight = weight.view(
        batch * conv.out_channel, in_channel, conv.kernel_size, conv.kernel_size
    )

    if conv.upsample:
        input = input.view(1, batch * in_channel, height, width)
        weight = weight.view(
            batch, conv.out_channel, in_channel, conv.kernel_size, conv.kernel_size
        )
        weight = weight.transpose(1, 2).reshape(
            batch * in_channel, conv.out_channel, conv.kernel_size, conv.kernel_size
        )
        out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, conv.out_channel, height, width)
        out = conv.blur(out)

    elif conv.downsample:
        input = conv.blur(input)
        _, _, height, width = input.shape
        input = input.view(1, batch * in_channel, height, width)
        out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, conv.out_channel, height, width)

    else:
        input = input.view(1, batch * in_channel, height, width)
        out = F.conv2d(input, weight, padding=conv.padding, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, conv.out_channel, height, width)
        
    out = layer.noise(out, noise=noise)
    out = layer.activate(out)
    
    return out

def decoder(G, style_space, latent, noise):

    out = G.input(latent)
    out = conv_warper(G.conv1, out, style_space[0], noise[0])
    skip = G.to_rgb1(out, latent[:, 1])

    i = 1
    for conv1, conv2, noise1, noise2, to_rgb in zip(
        G.convs[::2], G.convs[1::2], noise[1::2], noise[2::2], G.to_rgbs
    ):
        out = conv_warper(conv1, out, style_space[i], noise=noise1)
        out = conv_warper(conv2, out, style_space[i+1], noise=noise2)
        skip = to_rgb(out, latent[:, i + 2], skip)

        i += 2

    image = skip

    return image

def decoder_validate(G, style_space, latent):

    style_space = split_stylespace(style_space)
    noise = [getattr(G.noises, 'noise_{}'.format(i)) for i in range(G.num_layers)]

    out = G.input(latent)
    out = conv_warper(G.conv1, out, style_space[0], noise[0])
    skip = G.to_rgb1(out, latent[:, 1])

    i = 1
    for conv1, conv2, noise1, noise2, to_rgb in zip(
        G.convs[::2], G.convs[1::2], noise[1::2], noise[2::2], G.to_rgbs
    ):
        out = conv_warper(conv1, out, style_space[i], noise=noise1)
        out = conv_warper(conv2, out, style_space[i+1], noise=noise2)
        skip = to_rgb(out, latent[:, i + 2], skip)

        i += 2

    image = skip

    return image

def encoder_noise(G, noise):

    styles = [noise]
    style_space = []
    
    styles = [G.style(s) for s in styles]
    noise = [getattr(G.noises, 'noise_{}'.format(i)) for i in range(G.num_layers)]
    inject_index = G.n_latent
    latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
    style_space.append(G.conv1.conv.modulation(latent[:, 0]))

    i = 1
    for conv1, conv2, noise1, noise2, to_rgb in zip(
        G.convs[::2], G.convs[1::2], noise[1::2], noise[2::2], G.to_rgbs
    ):
        style_space.append(conv1.conv.modulation(latent[:, i]))
        style_space.append(conv2.conv.modulation(latent[:, i+1]))
        i += 2
        
    return style_space, latent, noise

def encoder_latent(G, latent):
    # an encoder warper for G

    style_space = []
    
    noise = [getattr(G.noises, 'noise_{}'.format(i)) for i in range(G.num_layers)]

    style_space.append(G.conv1.conv.modulation(latent[:, 0]))

    i = 1
    for conv1, conv2, to_rgb in zip(
        G.convs[::2], G.convs[1::2], G.to_rgbs
    ):
        style_space.append(conv1.conv.modulation(latent[:, i]))
        style_space.append(conv2.conv.modulation(latent[:, i+1]))
        i += 2
        
    return style_space, noise

def split_stylespace(style):
    style_space = []

    for idx in range(10):
        style_space.append(style[:, idx*512 : (idx+1) * 512])
    
    style_space.append(style[:, 10*512: 10*512 + 256])
    style_space.append(style[:, 10*512 + 256: 10*512 + 256*2])
    style_space.append(style[:, 10*512 + 256*2: 10*512 + 256*2 + 128])
    style_space.append(style[:, 10*512 + 256*2 + 128: 10*512 + 256*2 + 128 * 2])
    style_space.append(style[:, 10*512 + 256*2 + 128*2: 10*512 + 256*2 + 128*2 + 64])
    style_space.append(style[:, 10*512 + 256*2 + 128*2 + 64: 10*512 + 256*2 + 128*2 + 64*2])
    style_space.append(style[:, 10*512 + 256*2 + 128*2 + 64*2: 10*512 + 256*2 + 128*2 + 64*2 + 32])

    return style_space

def fuse_stylespace(style):
    new_s = torch.cat(style, dim=1)

    return new_s