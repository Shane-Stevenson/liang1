import torch
from torch.utils.data.dataset import Dataset
import tools.dataTools as dt
import numpy as np
import csv
import time

#Datasets for data loading into AIs

class VoxelGridMatrixDataset(Dataset):
    def __init__(self, voxels_in_csv_format : str, bound : int, height : int, voxel_size : float):
        file = open(voxels_in_csv_format)
        reader = csv.reader(file)
        voxels_in_csv_format = list(reader)
        voxels = []
        self.labels = []
        count = 0
        l = len(voxels_in_csv_format)
        for v in voxels_in_csv_format:
            (voxel, label) = dt.csv_to_voxel(v, bound, height, voxel_size)
            voxels.append(voxel)
            self.labels.append(label)
            count+=1
            print(count/l)

        self.voxel_xyz = []
        count = 0
        l = len(voxels)
        for v in voxels:
            # start = time.time()
            temp = (dt.voxel_to_zyx4D(v, bound, height, voxel_size))
            # end = time.time()
            # print('4D: ', end - start)
            # start = time.time()
            self.voxel_xyz.append(np.transpose(np.array(temp), (3, 1, 2, 0)))
            # end = time.time()
            # print('transpose: ', end - start)
            count+=1
            print(count/l)

        self.voxel_xyz = torch.FloatTensor((np.array(self.voxel_xyz)))

    def __len__(self):
        return len(self.voxel_xyz)
    
    def __getitem__(self, idx):
        return (self.voxel_xyz[idx], self.labels[idx])

class VoxelGridDataset(Dataset):
        def __init__(self, voxels_in_csv_format : str):
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
        def __init__(self, voxels_in_csv_format : str, length : int):
            reader = csv.reader(open(voxels_in_csv_format))
            self.voxels_in_csv_format = list(reader)
            self.voxels_in_csv_format = [[float(s) for s in row] for row in self.voxels_in_csv_format]
            self.labels = []
            count = 0
            l = len(self.voxels_in_csv_format)
            for v in self.voxels_in_csv_format:
                tempLabels = []
                while (len(v) != length):
                    tempLabels.append(v.pop())
                self.labels.append(tempLabels)
                count+=1
                print(count/l)
            
            self.voxels_in_csv_format = torch.FloatTensor(self.voxels_in_csv_format)
            self.labels = torch.FloatTensor(self.labels)

        def __len__(self):
            return len(self.voxels_in_csv_format)
    
        def __getitem__(self, idx):
            return (self.voxels_in_csv_format[idx], self.labels[idx])
        
class segmentationDataset2(Dataset):
        #Very poorly coded dataset for segmentation with multiple trees
        def __init__(self, voxels_in_csv_format : str, length : int):
            reader = csv.reader(open(voxels_in_csv_format))
            self.voxels_in_csv_format = list(reader)
            self.labels = []
            preLabels = []
            for v in self.voxels_in_csv_format:
                tempLabels = []
                while (len(v) != length):
                    tempLabels.append(eval(v.pop()))
                preLabels.append(tempLabels)

            for i in preLabels:
                j = len(i)-1
                temp = []
                while len(i) > 0:
                    t = i.pop(j)
                    temp.append(t[0])
                    temp.append(t[1])
                    j -= 1
                self.labels.append(temp)

            self.voxels_in_csv_format = [[float(s) for s in row] for row in self.voxels_in_csv_format]
            
            self.voxels_in_csv_format = torch.FloatTensor(self.voxels_in_csv_format)
            self.labels = torch.FloatTensor(self.labels)
            
            # print(self.labels)

        def __len__(self):
            return len(self.voxels_in_csv_format)
    
        def __getitem__(self, idx):
            return (self.voxels_in_csv_format[idx], self.labels[idx])