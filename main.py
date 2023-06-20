import geopandas as gpd
import numpy as np
import open3d as o3d
import tools.lasTools as lasTools
import tools.voxelization as voxelization
import tools.dataTools as dataTools
import AI.datasets as ds
import AI.ai1 as ai1
import AI.ai2 as ai2
import AI.AE1 as ae1
import AI.ai3 as ai3
import torch
import random
import json
import csv
import pandas as pd
import matplotlib.pyplot as plt

# data = ds.VoxelGridDataset('untTrees.csv', 30, 20, 1)
# loader = torch.utils.data.DataLoader(dataset=data, batch_size=4, shuffle=False) #batch size is 1 because weird dimension error???

# autoencoder = ae1.AE()

# autoencoder.train(loader, 6)


# for batch in loader:
#     reconstruction = autoencoder(batch)
#     print(len(reconstruction))
#     for i in batch:
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
#         print(new[0][0])
#         for z in range(20):
#             for x in range(30):
#                 for y in range(30):
#                     if new[z][x][y][0] > .25:
#                         points.append([x, y, z])
#                         colors.append([.9-z/20, .9-z/20, .9-z/20])
        
#         new = voxelization.voxelize(points, colors, 1, 30)
#         break

#     voxelization.visualize_voxel_grid(og)
#     voxelization.visualize_voxel_grid(new)