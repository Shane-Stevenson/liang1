import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(30*30*20, 4096) #First layer must be equal to the amount of voxels in a filled grid
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 128)
        self.fc7 = nn.Linear(128, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)

        return F.sigmoid(x) * 30
    
    def train(self, DataLoader : torch.utils.data.DataLoader, EPOCHS : int):
        optimizer = optim.Adam(self.parameters(), lr=.0003)
        for epoch in range(EPOCHS):
            count = 0
            totalLoss = 0
            batchLoss = 0
            l = len(DataLoader)
            for batch in DataLoader:
                data, labels = batch
                output = self(data.to('mps'))

                n = np.zeros(labels.size())
                adjLabels = torch.FloatTensor(n)
                for i, tens in enumerate(labels):
                    for j,val  in enumerate(tens):
                        adjLabels[i][j] = val # Bound

                #Then deal with prediction strength

                print(adjLabels)
                print(output)
                batchLoss = F.mse_loss(adjLabels.to('mps'), output.to('mps'))
                optimizer.zero_grad()
                batchLoss.backward()
                # for name, param in self.named_parameters():
                #     print(name, param.grad)
                optimizer.step()
                print(batchLoss)
                totalLoss += batchLoss.item()
                count+=1

                # print(count/l)
                print()
            
            # print('avg loss: ', totalLoss/count)
                

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

def getBoxesToKeep(boxes : torch.FloatTensor, confidence : torch.FloatTensor, threshold : float):
    keep = {i for i in range(0, len(boxes))}
    JM = jaccard(boxes, boxes)
    for i in range(len(JM)):
        for j in range(len(JM[i])):
            if i == j: continue
            if(JM[i][j] > threshold):
                print(keep)
                if confidence[i] > confidence[j]:
                    if i in keep:
                        keep.remove(i)
                else:
                    if j in keep:
                        keep.remove(j)
    return [i for i in keep]

class Net2(nn.Module):
    """
    Segmentation AI that implements NMS
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(30*30*20, 4096) #First layer must be equal to the amount of voxels in a filled grid
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 128)
        self.fc7 = nn.Linear(128, 25)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)

        #Muliply output by 30 to match size of voxel grid. Needs to change based on the bounds of the voxel grid to be processed
        return F.sigmoid(x) * 30
    
    def train(self, DataLoader : torch.utils.data.DataLoader, EPOCHS : int):

        optimizer = optim.Adam(self.parameters(), lr=.0003)
        for epoch in range(EPOCHS):
            count = 0
            totalLoss = 0
            batchLoss = 0
            for batch in DataLoader:
                data, labels = batch
                output = self(data.to('mps'))
                
                #This chunk of code separates the bounding boxes coordinates and their respective scores for Non-Max Supression
                boxes = output.reshape(-1,5)[:,:4]
                boxes = boxes.reshape(len(data),20)
                boxes = boxes.reshape(len(data),5,4)
                scores = output.reshape(-1,5)[4::5]
                scores = scores.reshape(len(data), 5)
                #Scores are divided by 30 to be between 0 and 1
                scores = scores / 30
                
                adjBoxes = torch.zeros_like(boxes)
                for i in range(len(boxes)):
                    #The 'toAdd' tensor is to ensure that at the beginning of training NMS can function ensuring that x1 < x2 and y1 < y2 for each box
                    toAdd = torch.zeros_like(boxes[i])
                    for j in range(len(boxes[i])):
                        toAdd[j][0] = 0
                        toAdd[j][1] = 0
                        toAdd[j][2] = 2
                        toAdd[j][3] = 2
                    #getBoxesToKeep is my implementation of Non-Max supression
                    keepBoxes = getBoxesToKeep(boxes[i]+toAdd, scores[i], .1)
                    keepBoxes.sort()
                    #idxToZero contains all the bounding boxes that will be thrown out via NMS
                    idxToZero = [j for j in range(0, 5) if j not in keepBoxes]
                    #A mask is used to zero out values and ensure backpropogation will still work
                    mask = torch.ones_like(boxes[i])
                    for j in idxToZero:
                        mask[j] = 0
                    adjBoxes[i] = torch.mul(mask, boxes[i])

                    #Here I need to implement the switching of bounding boxes (Maybe base on size?)

                    print('----->', idxToZero)
                    print('----->', scores)
                    print('labels ->', labels)
                    print('prediction ->', adjBoxes)

                #Data is reshaped for the loss function
                adjBoxes = adjBoxes.reshape(len(data),20)

                batchLoss = F.mse_loss(labels.to('mps'), adjBoxes.to('mps'))
                optimizer.zero_grad()
                batchLoss.backward()
                optimizer.step()
                print('loss: ', batchLoss.item())
                totalLoss += batchLoss.item()
                count+=1

                # print(count/l)
                print()
            print('avg loss :', totalLoss/count)
            # print('avg loss: ', totalLoss/count)