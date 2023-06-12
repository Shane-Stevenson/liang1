import torch
from torch.utils.data.dataset import Dataset
import tools.dataTools as dt
import numpy as np
import csv

class VoxelGridDataset(Dataset):
    def __init__(self, voxels_in_csv_format : list, bound : int, height : int, voxel_size : float):
        file = open(voxels_in_csv_format)
        reader = csv.reader(file)
        voxels_in_csv_format = list(reader)
        voxels = []
        for v in voxels_in_csv_format:
            voxels.append(dt.csv_to_voxel(v, bound, height, voxel_size)[0])

        self.voxel_zyx = []
        for v in voxels:
            self.voxel_zyx.append(dt.voxel_to_zyx(v, bound, height, voxel_size))

        self.voxel_zyx = np.array(self.voxel_zyx)

    def __len__(self):
        return len(self.voxel_zyx)
    
    def __getitem__(self, idx):
        return self.voxel_zyx[idx]

        