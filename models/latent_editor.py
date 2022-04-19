import torch
from utils.utils import tensor2im


class LatentEditor(object):

    def __init__(self,opts):
        #boundries trained by anycost-gan
        directions = torch.load(opts.directions)
        self.interfacegan_directions = {("_".join(key.split("_")[1:])).lower():directions[key].cuda() for key in directions.keys()}
        # self.interfacegan_directions['age'] = torch.load('/home/xujiamu/Coding/FaceEdit/restyle-encoder/editing/interfacegan_directions/age.pt').cuda()
        self.opts = opts

    # (n,1,18,512) (1,512)
    def apply_interfacegan(self, latents, direction,factor=None):
        direction_code = self.interfacegan_directions[direction]
        factor = self.opts.directions_factor if factor is None else factor
        edit_latents = latents + factor*direction_code
        return edit_latents


