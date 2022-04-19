import torch
import torch.nn as nn
from models.e4e import e4e
from models.latent_editor import LatentEditor
from models.attribute_classifier import BranchedTinyAttr
from argparse import Namespace

class Train_Model(nn.Module):
    def __init__(self,configs,device='cpu'):
        super().__init__()
        ckpt = torch.load(configs.checkpoint_path, map_location='cpu')
        opts = ckpt['opts']
        opts['checkpoint_path'] = configs.checkpoint_path
        opts = Namespace(**opts)
        self.encoder = e4e(opts)
        self.generator = self.encoder.decoder
        self.classifier = BranchedTinyAttr(configs.classifier)
        self.classifier.set_idx(configs.attribute)
        self.latent_editor = LatentEditor(configs)
        self.eval()
        self.to(device)

class Eval_Model(nn.Module):
    def __init__(self,configs,device='cpu'):
        super().__init__()
        ckpt = torch.load(configs.checkpoint_path, map_location='cpu')
        opts = ckpt['opts']
        opts['checkpoint_path'] = configs.checkpoint_path
        opts = Namespace(**opts)
        self.encoder = e4e(opts)
        self.generator = self.encoder.decoder
        self.latent_editor = LatentEditor(configs)
        self.eval()
        self.to(device)
