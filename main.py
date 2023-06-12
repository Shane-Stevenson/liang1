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

data = ds.VoxelGridDataset('untTreesTrain.csv', 30, 20, 1)
print(len(data))
print(data[3])
loader = torch.utils.data.DataLoader(dataset=data, batch_size=4, shuffle=False)

autoencoder = ae1.AE()
zxy = data[0].tolist()
points = []
colors = []

for z in range(20):
    for x in range(30):
        for y in range(30):
            if zxy[z][x][y] == 1:
                points.append([x, y, z])
                colors.append([.9-z/20, .9-z/20, .9-z/20])

v1 = voxelization.voxelize(points, colors, 1, 30)
# voxelization.visualize_voxel_grid(v1)

for i in loader:
    b = autoencoder(torch.Tensor(i))

# count = 0
# (voxels, xy) = lasTools.iterate_over_las('data/unt2.las', 30, 1)

# zipped = list(zip(voxels, xy))
# random.shuffle(zipped)
# voxels, xy = zip(*zipped)

# for i in range(len(xy)):
#     if xy[i][0] >  671928.15 and xy[i][2] < 672826.5200000001 and xy[i][1] > 3675742.49 and xy[i][3] < 3675893.22:
#         print("(", xy[i][0], ", ", xy[i][1], "), (", xy[i][2], ", ", xy[i][3], ")")
#         voxelization.visualize_voxel_grid(voxels[i])
#         usr = input('How many trees are contained in that voxel grid?')
#         c = voxelization.fill_voxel_grid(voxels[i], 30, 20, 1)[1]
#         dataTools.voxel_to_csv('untTreesTrain.csv', c, usr)

# dataTools.affine_augment_csv_file('untTreesTrain.csv', 'untAffineTrain.csv', 30, 20, 1)
# dataTools.random_augment_csv('untAffineTrain.csv', 'untRandomTrain.csv', 30, 20, 1, 1, 4)
# net = ai3.Net()

# net.train('untRandomTrain.csv', 3)
# net.test('untTreesTrain.csv')
# net.test('untAffineTrain.csv')
# net.test('untRandomTrain.csv')
# print('-------------------------------------------')
# net.test('untTrees.csv')

# file = open('untTrees.csv')
# reader = csv.reader(file)
# l = list(reader)
# print(len(l))

# files = ['training/regression/30Train.csv', 'training/regression/ky1.csv', 'training/regression/og1.csv']

# merge = open('training/regression/net.csv', 'w')
# writer = csv.writer(merge)

# for i in files:
#     file = open(i)
#     reader = csv.reader(file)
#     voxels = list(reader)
#     for j in voxels:
#         writer.writerow(j)