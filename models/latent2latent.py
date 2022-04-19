import torch.nn as nn
from models.stylegan2.model import EqualLinear, PixelNorm
import torch

class Mapper(nn.Module):
    def __init__(self,latent_dim=512):
        super(Mapper,self).__init__()
        layers = []
        for i in range(2):
            layers.append(
                nn.Sequential(
                    EqualLinear(latent_dim, latent_dim, lr_mul=0.01),
                    nn.Tanh(),
                )
            )
        layers.append(nn.Sequential(EqualLinear(latent_dim, latent_dim, lr_mul=0.01)))
        self.mapping = nn.Sequential(*layers)

    def forward(self, x):
        x = self.mapping(x)
        return x

class Latent2Latent(nn.Module):
    def __init__(self):
        super(Latent2Latent,self).__init__()
        self.course_mapping = Mapper()
        self.medium_mapping = Mapper()
        self.fine_mapping = Mapper()

    def forward(self, x):
        x_coarse = x[:, :, :4, :]
        x_medium = x[:, :, 4:8, :]
        x_fine = x[:, :, 8:, :]
        x_coarse = self.course_mapping(x_coarse)
        x_medium = self.medium_mapping(x_medium)
        x_fine = self.fine_mapping(x_fine)
        out = torch.cat([x_coarse, x_medium, x_fine], dim=2)

        return out