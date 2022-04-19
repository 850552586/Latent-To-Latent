from argparse import Namespace
from tqdm import tqdm
import time
import numpy as np
import torch
from PIL import Image

def get_latent(input,net,opt):
    generator = net.decoder
    with torch.no_grad():
        input_cuda = input.cuda().float()
        avg_image = get_average_image(net)
        y_hat, latent = None, None
        for iter in range(opt.rsencoder_n_iters):
            if iter == 0:
                avg_image_for_batch = avg_image.unsqueeze(0).repeat(input_cuda.shape[0], 1, 1, 1)
                x_input = torch.cat([input_cuda, avg_image_for_batch], dim=1)
            else:
                x_input = torch.cat([input_cuda, y_hat], dim=1)
            y_hat, latent = net.forward(x_input,
                                    latent=latent,
                                    randomize_noise=False,
                                    return_latents=True,
                                    resize=True)
        g_imgs = y_hat #(n,3,w,h)
        w_latents = torch.stack([latent]).transpose(0,1)
        return g_imgs,w_latents


def get_average_image(net):
    avg_image = net(net.latent_avg.unsqueeze(0),
                    input_code=True,
                    randomize_noise=False,
                    return_latents=False,
                    average_code=True)[0]
    avg_image = avg_image.to('cuda').float().detach()
    return avg_image