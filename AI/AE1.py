import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import csv

#Autoencoder that never really worked

class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encode1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(4,4,4), stride=1, padding=(0,0,0))
        self.encode2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(3,3,3), stride=1, padding=(0,0,0))
        self.encode3 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3,3,3), stride=1, padding=(0,0,0))
        self.encode4 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3,3,3), stride=1, padding=(0,0,0))
        self.decode1 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=(3,3,3), stride=1, padding=(0,0,0))
        self.decode2 = nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=(3,3,3), stride=1, padding=(0,0,0))
        self.decode3 = nn.ConvTranspose3d(in_channels=16, out_channels=8, kernel_size=(3,3,3), stride=1, padding=(0,0,0))
        self.decode4 = nn.ConvTranspose3d(in_channels=8, out_channels=1, kernel_size=(4,4,4), stride=1, padding=(0,0,0))

    def forward(self, x):
        latent = self.encode1(x)
        latent = nn.BatchNorm3d(8)(latent)
        latent = nn.ReLU()(latent)
        latent = self.encode2(latent)
        latent = nn.BatchNorm3d(16)(latent)
        latent = nn.ReLU()(latent)
        latent = self.encode3(latent)
        latent = nn.BatchNorm3d(32)(latent)
        latent = nn.ReLU()(latent)
        latent = self.encode4(latent)
        latent = nn.BatchNorm3d(64)(latent)
        latent = nn.ReLU()(latent)
        print(latent.size())
        y = self.decode1(latent)
        y = nn.BatchNorm3d(32)(y)
        y = nn.ReLU()(y)
        y = self.decode2(y)
        y = nn.BatchNorm3d(16)(y)
        y = nn.ReLU()(y)
        y = self.decode3(y)
        y = nn.BatchNorm3d(8)(y)
        y = nn.ReLU()(y)
        y = self.decode4(y)
        y = nn.BatchNorm3d(1)(y)
        y = nn.ReLU()(y)
        y = nn.Sigmoid()(y)
        
        return y

    def train(self, DataLoader : torch.utils.data.DataLoader, EPOCHS : int):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        for epoch in range(EPOCHS):
            count = 0
            totalLoss = 0
            for batch in DataLoader:
                reconstruction = self(batch[0])
                loss = F.binary_cross_entropy(reconstruction, batch[0])
                totalLoss += loss.item()
                count +=1
                loss.backward()
                optimizer.step()
                # print(count/len(DataLoader))
            
            print('Avg loss : ', totalLoss/count)
            

##########################################################
#scrapped code??

# print('dataset')
# data = ds.VoxelGridDataset('train.csv', 30, 20, 1)
# print('loader')
# loader = torch.utils.data.DataLoader(dataset=data, batch_size=4, shuffle=False) 
# print('initializing AE')
# autoencoder = ae1.AE()
# print('training')
# autoencoder.train(loader, 8)
# # autoencoder.load_state_dict(torch.load('s'))
# # autoencoder.train(loader, 1)

# torch.save(autoencoder.state_dict(), 's')

# for batch in loader:
#     reconstruction = autoencoder(batch[0])
#     for i in batch[0]:
#         points = []
#         colors = []
#         og = np.transpose(i, (3, 1, 2, 0))
#         for z in range(20):
#             for x in range(30):
#                 for y in range(30):
#                     if og[z][x][y][0] == 1:
#                         points.append([x, y, z])
#                         colors.append([.9-z/20, .9-z/20, .9-z/20])
        
#         og = voxelization.voxelize(points, colors, 1, 30)
#         break

#     for i in reconstruction:
#         points = []
#         colors = []
#         new = np.transpose(i.detach().numpy(), (3, 1, 2, 0))
#         for z in range(20):
#             for x in range(30):
#                 for y in range(30):
#                     if new[z][x][y][0] > .55:
#                         points.append([x, y, z])
#                         colors.append([.9-z/20, .9-z/20, .9-z/20])
        
#         new = voxelization.voxelize(points, colors, 1, 30)
#         break

#     voxelization.visualize_voxel_grid(og)
#     voxelization.visualize_voxel_grid(new)