from utils.face_extractor import crop_align
import sys
sys.path.append(".")
sys.path.append("..")
import os
from utils.latent_utils import get_latent
from PIL import Image
import torch
import yaml
from easydict import EasyDict
from utils.utils import tensor2im,interpolate
from argparse import ArgumentParser
from torch import nn
import numpy as np
from training.train_model import Eval_Model
from models.latent2latent import Latent2Latent
from tqdm import tqdm
import torchvision.transforms as transforms
import time

class FaceEditor:
    def __init__(self,args,device="cuda"):
        self.configs = self.init_config(args)
        self.inference_transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.latent_model = Latent2Latent()
        ckpt = torch.load(args.ckpt)
        self.latent_model.load_state_dict(ckpt)
        self.latent_model.to(device)
        self.model = Eval_Model(self.configs,device)
        self.device = device
    
    def init_config(self,args):
        with open("./configs.yaml",encoding='utf8')as f:
            configs = yaml.load(f.read(), Loader=yaml.FullLoader)
        configs = EasyDict(configs)
        configs.attribute = args.attribute
        configs.exdir = args.exdir
        configs.savedir = args.savedir
        return configs

    def face_align(self,imgpath)->str:
        return crop_align(imgpath,"{}/tmp".format(self.configs.exdir))

    def edit(self,imgpath):
        align_img = self.face_align(imgpath)
        img = Image.open(align_img).convert('RGB')
        img = self.inference_transform(img).unsqueeze(0)
        with torch.no_grad():
            _,w_latents = get_latent(img,self.model.encoder,self.configs)
            w_a_latents = self.model.latent_editor.apply_interfacegan(w_latents,self.configs.attribute) # add original attributes to latent code w
            w_n = self.latent_model(w_a_latents)
            w_n = 0.7*w_latents + 0.3*w_n
            g_img = self.model.generator(w_n, randomize_noise=False, input_is_latent=True)[0]
            r_img = np.array(tensor2im(g_img[0]))
            img = Image.fromarray(r_img)
            imgid = int(time.time())
            img.save("{}/{}.png".format(self.configs.savedir,imgid))
        print("------inference finished---------------")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--attribute', type=str,default=None,help='attribute for face-edit inference')
    parser.add_argument('--gpu', type=bool, default=None)
    parser.add_argument('--ckpt', type=str, default=None,help='attribute ckpt for face-edit inference')
    parser.add_argument('--exdir', type=str, default=None,help='experiment path for face-edit inference')
    parser.add_argument('--savedir', type=str, default=None,help='save path for face-edit inference')
    args = parser.parse_args()
    faceeditor = FaceEditor(args)
    faceeditor.edit("test/test.jpg")

            