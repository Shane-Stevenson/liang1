import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tools.dataTools as ds
import tools.segmentedUtil as su
import tools.voxelization as voxelization
import torchvision
import numpy as np
import torch.nn.init as init

"""
This file is the final iteration of our AI that used 3D convolutional Layers. We realized eventually
that even with a proper loss function for this AI, we would require far more data than we initially had and thus
decided to abandon the custom AI strategy.
"""

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

def NMS(input : torch.FloatTensor, threshold : float):
    #This chunk of code separates the bounding boxes coordinates and their respective scores for Non-Max Supression
    boxes = input.reshape(-1,5)[:,:4]
    # boxes = boxes.reshape(batchsize,20)
    boxes = boxes.reshape(len(boxes),4)
    scores = input.reshape(-1)[4::5]
    scores = scores.reshape(len(boxes))
    #Scores are divided by 30 to be between 0 and 1
    scores = scores / 30
    # print(boxes)
    # quit()

    # print(boxes)
    # print(scores)

    # The 'toAdd' tensor is to ensure that at the beginning of training NMS can function ensuring that x1 < x2 and y1 < y2 for each box
    toAdd = torch.zeros_like(boxes)
    for j in range(len(boxes)):
        toAdd[j][0] = -2
        toAdd[j][1] = -2
        toAdd[j][2] = 2
        toAdd[j][3] = 2
    adjBoxes = torch.zeros_like(boxes)
    adjBoxes = torch.add(boxes, toAdd)
    # adjBoxes = boxes
    keepBoxes = nmsUtil(adjBoxes, scores, threshold)
    adjBoxes = adjBoxes[keepBoxes]
    scores = scores[keepBoxes]
    return adjBoxes, scores

def nmsUtil(boxes : torch.FloatTensor, confidence : torch.FloatTensor, threshold : float):
    # print(boxes)
    keep = {i for i in range(0, len(boxes))}
    JM = jaccard(boxes, boxes)
    # print(boxes)
    # print(confidence)
    # print(JM)
    for i in range(len(JM)):
        for j in range(len(JM[i])):
            if i == j: continue
            if(JM[i][j] > threshold):
                if confidence[i] > confidence[j]:
                    if j in keep:
                        keep.remove(j)
                else:
                    if i in keep:
                        keep.remove(i)
    for i in range(len(boxes)):
        if boxes[i][0] > boxes[i][2] or boxes[i][1] > boxes[i][3]:
            if i in keep:
                keep.remove(i)
    ret = [i for i in keep]
    ret = ret.sort(key=lambda x:confidence[x] )
    # quit()
    return [i for i in keep]

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        # Define any additional parameters or initialization logic here

    def forward(self, predictions, labels, scores):
        """
        This is a mostly unfinished attempt at implementing YOLO's loss function.
        """
        trueScores = torch.ones_like(scores)
        if labels.size()[0] > predictions.size()[0]:
            diff = labels.size()[0] - predictions.size()[0]
            labels = labels[: predictions.size()[0]]
        elif labels.size()[0] < predictions.size()[0]:
            t = len(scores)-1
            for i in range(predictions.size()[0]-labels.size()[0]):
                scores[t] = 0
                t-=1
            diff = labels.size()[0] - predictions.size()[0]
            predictions = predictions[: labels.size()[0]]
        else:
            diff = 0
        
        trueScores = torch.ones_like((scores))
        JM = jaccard(labels, predictions)
        # print(JM)

        seen = set()
        for k in range(len(JM)):
            indices = [i for i in range(labels.size()[0])]
            m = 0
            mIndex = ()

            for idx1, i in enumerate(JM):
                if idx1 in seen:
                    continue
                for idx2, j in enumerate(i):
                    if idx2 in seen:
                        continue
                    if j.item() >= m:
                        m = j
                        mIndex = (idx1, idx2)

            seen.add(mIndex[1])
            indices[mIndex[0]], indices[mIndex[1]] = indices[mIndex[1]], indices[mIndex[0]]
            labels = labels[indices]
            JM = JM[indices]

        print(labels)
        print(predictions)
        # quit()
        print(JM)

        loss = F.l1_loss(labels.to('mps'), predictions.to('mps'))
        loss.add(F.cross_entropy(scores, trueScores))
        if diff != 0:
            loss = loss.add((diff*4))
        # if diff != 0:
        #     loss = loss.mul((diff+1))

        # loss = torch.Tensor([diff*50])
        return loss

class Net(nn.Module):
    """
    Segmentation AI that implements NMS
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=4, kernel_size=(4,4,4), stride=1, padding=(0,0,0))
        self.conv2 = nn.Conv3d(in_channels=4, out_channels=8, kernel_size=(4,4,4), stride=1, padding=(0,0,0))
        self.conv3 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(4,4,4), stride=1, padding=(0,0,0))
        self.conv4 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(4,4,4), stride=1, padding=(0,0,0))
        self.conv5 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(4,4,4), stride=1, padding=(0,0,0))
        self.conv6 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(4,4,4), stride=1, padding=(0,0,0))
        self.fc1 = nn.Linear(25088, 4096)
        self.fc2 = nn.Linear(4096, 512)
        # self.fc3 = nn.Linear(2048, 1024)
        # self.fc4 = nn.Linear(1024, 1024)
        # self.fc5 = nn.Linear(1024, 512)
        # self.fc6 = nn.Linear(512, 512)
        self.fc7 = nn.Linear(512, 250)

        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight, gain = 25)
        # init.xavier_uniform_(self.fc3.weight)
        # init.xavier_uniform_(self.fc4.weight)
        # init.xavier_uniform_(self.fc5.weight)
        # init.xavier_uniform_(self.fc6.weight)
        init.xavier_uniform_(self.fc7.weight, gain = 25)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = nn.MaxPool3d(4, 1, 0)(x)
        x = F.relu(self.conv2(x))
        x = nn.MaxPool3d(4, 1, 0)(x)
        x = F.relu(self.conv3(x))
        x = nn.MaxPool3d(4, 1, 0)(x)
        x = F.relu(self.conv4(x))
        x = nn.MaxPool3d(4, 1, 0)(x)
        x = F.relu(self.conv5(x))
        x = nn.MaxPool3d(4, 1, 0)(x)
        x = F.relu(self.conv6(x))
        x = nn.MaxPool3d(4, 1, 0)(x)
        x = torch.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))
        # x = F.relu(self.fc6(x))
        # x = self.fc7(x)
        x = F.sigmoid(x) * 25

        x = NMS(x, .05)

        return x
    
    def train(self, DataLoader : torch.utils.data.DataLoader, EPOCHS : int):
        """
        Standard training procedures. Fetch data, pass through AI, calculate loss, backpropogation.
        """
        optimizer = optim.Adam(self.parameters(), lr=.00003)
        netCount = 0
        for epoch in range(EPOCHS):
            count = 0
            totalLoss = 0
            for batch in DataLoader:
                data, labels = batch
                output, scores= self(data)
                labels = labels.reshape(labels.size()[1],4)

                L = CustomLoss()
                loss = L(output.to('mps'), labels.to('mps'), scores.to('mps'))
                print('loss: ', loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                totalLoss += loss.item()
                count+=1
                netCount+=1

                print(count/len(DataLoader))
                print()

    def test(self, DataLoader : torch.utils.data.DataLoader):
        for batch in DataLoader:
            data, label = batch
            label.to('mps')
            output = self(data.to('mps'))

            print(label)
            print(output)
            # print(jaccard(label, output))