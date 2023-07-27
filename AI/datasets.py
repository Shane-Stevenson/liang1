import torch
from torch.utils.data.dataset import Dataset
import tools.dataTools as dt
import numpy as np
import sys
import csv
import time
csv.field_size_limit(sys.maxsize)

"""
This file contains the various custom datasets we created to pass Data into our AI using PyTorch's dataloaders
"""

class VoxelGridMatrixDataset(Dataset):
    """
    This dataset is to be used for convolutional AIs
    """
    def __init__(self, voxels_in_csv_format : str, bound : int, height : int, voxel_size : float):
        file = open(voxels_in_csv_format)
        reader = csv.reader(file)
        voxels_in_csv_format = list(reader)
        voxels = []
        self.labels = []
        count = 0
        l = len(voxels_in_csv_format)
        #Here we fill voxels and self.labels with the labeled data
        for v in voxels_in_csv_format:
            (voxel, label) = dt.csv_to_voxel(v, bound, height, voxel_size)
            voxels.append(voxel)
            self.labels.append(label)
            count+=1
            print(count/l)

        self.voxel_xyz = []
        count = 0
        l = len(voxels)
        #Here the voxels are turned into 4D PyTorch Tensors for compatibility with 3dconv
        for v in voxels:
            temp = (dt.voxel_to_zyx4D(v, bound, height, voxel_size))
            self.voxel_xyz.append(np.transpose(np.array(temp), (3, 1, 2, 0)))
            count+=1
            print(count/l)

        self.voxel_xyz = torch.FloatTensor((np.array(self.voxel_xyz)))

    def __len__(self):
        return len(self.voxel_xyz)
    
    def __getitem__(self, idx):
        return (self.voxel_xyz[idx], self.labels[idx])

class VoxelGridDataset(Dataset):
        """
        First iteration of a dataset for regression
        """
        def __init__(self, voxels_in_csv_format : str):
            #Fill self.labels and self.voxels_in_csv_format with our labeled data
            reader = csv.reader(open(voxels_in_csv_format))
            self.voxels_in_csv_format = list(reader)
            self.voxels_in_csv_format = [[float(s) for s in row] for row in self.voxels_in_csv_format]
            self.labels = []
            count = 0
            l = len(self.voxels_in_csv_format)
            for v in self.voxels_in_csv_format:
                self.labels.append(v.pop())
                count+=1
                print(count/l)
            
            self.voxels_in_csv_format = torch.FloatTensor(self.voxels_in_csv_format)
            self.labels = torch.FloatTensor(self.labels)

        def __len__(self):
            return len(self.voxels_in_csv_format)
    
        def __getitem__(self, idx):
            return (self.voxels_in_csv_format[idx], self.labels[idx])
        
class segmentationDataset(Dataset):
        """
        This is the dataset for our final segmentation model
        """
        def __init__(self, voxels_in_csv_format : str, bound : int, height : int, voxel_size : float):
            #Read the data
            reader = csv.reader(open(voxels_in_csv_format))
            initVoxels = list(reader)
            self.labels = []

            #Remove the bounding boxes from the data and add them to this dataset's labels
            for v in initVoxels:
                tempLabels = []
                while (len(v) != 1):
                    tempLabels.append(eval(v.pop()))
                self.labels.append(tempLabels)

            #Append the voxel data
            self.voxels_in_csv_format = []
            for v in initVoxels:
                tmp = eval(v[0])
                tmp = np.array(a)
                tmp = tmp.reshape(1, int(bound/voxel_size), int(bound/voxel_size), int(height/voxel_size))
                self.voxels_in_csv_format.append(tmp)
            
            self.voxels_in_csv_format = torch.FloatTensor(np.array(self.voxels_in_csv_format))
            for i in range(len(self.labels)):
                self.labels[i] = torch.FloatTensor(self.labels[i])
            
            # print(self.labels)

        def __len__(self):
            return len(self.voxels_in_csv_format)
    
        def __getitem__(self, idx):
            return (self.voxels_in_csv_format[idx], self.labels[idx])