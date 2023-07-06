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
import AI.ai4 as ai4
import AI.treeSegmentAI as segmentAI
import torch
import random
import json
import csv
import fiona
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision
def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]

def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]
# lasTools.segment_over_las_with_shp('data/unt2.las', 'data/tempshape.gpkg', 'z.csv', 671899.17, 3675854.3000000003, 672015.4900000001, 3676022.1300000004, 30, 20, 1)

# pred = torch.FloatTensor([[14.5634, 13.7611, 20.4098, 17.4150]])
# real = torch.FloatTensor([[ 0., 12.,  5., 22]])

# pred2 = torch.FloatTensor([[14.5719, 15.5502, 17.1706, 17.2630]])
# real2 = torch.FloatTensor([[15.2628, 15.2623, 17.0816, 16.4977]])
# print(jaccard(pred2, real2))
# # print(jaccard(pred, real))
# quit()

dataset = ds.segmentationDataset2('zzz.csv', 18000)

loader = DataLoader(dataset, 1, False)

net = segmentAI.Net2()
net.to('mps')
net.train(loader, 30)

#---------------------------------------
# truth = torch.FloatTensor(np.array([4, 21, 9, 25, 21, 25, 27, 27]))
# x = torch.FloatTensor(np.array([[4, 20, 8, 24], [5, 21, 9, 25], [3, 19, 7, 23], [20, 24, 28, 27], [18, 25, 29, 27]]))
# x.requires_grad = True
# conf = torch.FloatTensor(np.array([.8, .7, .5, .9, .6]))
# idxs = torchvision.ops.nms(x, conf, .2)
# idxs = idxs.sort(descending=False)[0]
# res1 = x[idxs]
# result = torch.flatten(res1)
# # result = torch.FloatTensor(np.ndarray.flatten(np.array([x[j] for j in idxs])))

# print(result)
# print(truth)

# loss = torch.nn.functional.mse_loss(result, truth)
# loss.backward()
# print(loss)
#---------------------------------------

# net = segmentAI.Net()
# net.to('mps')

# dataset = ds.segmentationDataset('segmented.csv', 18000)
# loader = DataLoader(dataset, batch_size=1, shuffle=False)
# loader

# net.train(loader, 100)