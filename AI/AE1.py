import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import csv

class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(4,4,4), stride=1, padding=(1,1,1)),
            print('ooo'),
            nn.BatchNorm3d(num_features=8),
            nn.ReLU(),
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(3,3,3), stride=1, padding=(1,1,1)),
            print('eeeee'),
            nn.BatchNorm3d(num_features=16),
            nn.ReLU(),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            print('aaaaa'),
            nn.BatchNorm3d(num_features=32),
            nn.ReLU(),
            print('what')
        )

        self.decoder = nn.Sequential(
            nn.BatchNorm3d(num_features=32),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=16),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(num_features=8),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(num_features=1),
            nn.ReLU()
        )

    def forward(self, x):
        print('forward')
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction
