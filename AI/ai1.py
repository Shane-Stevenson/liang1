
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
import torchvision
from torchvision import transforms, datasets
import tools.dataTools as aaa
import tools.voxelization as voxelization

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8000, 1024) #First layer must be equal to the amount of voxels in a filled grid
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 16)
        self.fc5 = nn.Linear(16, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)

        return F.softmax(x, dim = -1)
    
def trainAndTest(data : str):
    # open and read data
    writer = SummaryWriter()
    csvreader = csv.reader(open(data))
    train = list(csvreader)
    print(len(train))
    print(len(train[0]))
    

    #change to float
    train = [[float(s) for s in row] for row in train]

    #create test list for output comparation
    test = []
    for x in range(len(train)):
        test.append(train[x][len(train[x])-1])
        train[x].pop(len(train[x])-1) #pop the output off the train data

    #convert to tensors
    tensor_train = torch.Tensor(train)
    tensor_test = torch.Tensor(test)
    print(tensor_test)

    #separate data set into training and testing
    x_train, x_test, y_train, y_test = train_test_split(tensor_train, tensor_test, test_size=.2, shuffle=False)
    x_train = x_train.type(torch.FloatTensor)
    x_test = x_test.type(torch.FloatTensor)
    y_train = y_train.type(torch.FloatTensor)
    y_test = y_test.type(torch.FloatTensor)

    batchSize = 8

    net = Net()

    optimizer = optim.Adam(net.parameters(), lr = .00003)
    EPOCHS = 6

    step = 0

    #train
    for epoch in range(EPOCHS): 
        random = torch.randperm(x_train.shape[0])
        for j in range(x_train.shape[0]):
            i = random[j]
            if y_train[i] == 0:
                y = torch.tensor([1., 0., 0.])
            elif y_train[i] == 1:
                y = torch.tensor([0., 1., 0.])
            else:
                y = torch.tensor([0., 0., 1.])
            output = net(x_train[i])
            loss = F.binary_cross_entropy(output, y) 
            writer.add_scalar("loss/train", loss, step)
            step += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(loss)

    step = 0
    correct = 0
    label = 0
    aiResult = 0

    AInoTree = 0
    AItree = 0
    AImultipleTree = 0

    noTree = 0
    tree = 0
    multipleTree = 0


    total = 0

    #test
    random = torch.randperm(x_test.shape[0])
    with torch.no_grad():
        for j in range(x_test.shape[0]):

            i = random[j]
            label = y_test[i]

            output = net(x_test[i])
            loss = F.binary_cross_entropy(output, y)
            writer.add_scalar("loss/test", loss, step)
            step += 1

            if max(output[0], output[1], output[2]) == output[0]:
                aiResult = 0
                AInoTree += 1
            elif max(output[0], output[1], output[2]) == output[1]:
                aiResult = 1
                AItree += 1
            else:
                aiResult = 2
                AImultipleTree += 1

            if label == 0:
                noTree += 1
            elif label == 1:
                tree += 1
            else:
                multipleTree += 1

            #output[0] > output[1] represents the AI saying it is not a tree
            if label == aiResult:
                correct+=1
                print('right: ', output, '(label = ', label, ')')
            else:
                # v = aaa.csv_to_voxel(x_test[i].tolist(), 20, 1)[0]
                # print('output: ', aiResult)
                # print('label: ', label)
                # voxelization.visualize_voxel_grid(v)
                print('wrong: ', output, '(label = ', label, ')')

            total += 1
        print(loss)
    
    print("Accuracy: ", correct/total)
    print('no tree: ', AInoTree, ' : ', noTree)
    print('tree: ', AItree, ' : ',tree)
    print('multiple tree: ', AImultipleTree, ' : ',multipleTree)