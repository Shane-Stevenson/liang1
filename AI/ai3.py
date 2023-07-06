import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(30*30*20, 2048) #First layer must be equal to the amount of voxels in a filled grid
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        # self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 128)
        self.fc7 = nn.Linear(128, 7)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)

        return F.softmax(x, dim = -1)
    
    def train(self, data : str, epochs : int):
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
        tensor_train
        tensor_test
        print(tensor_test)

        #separate data set into training and testing
        x_train = tensor_train
        y_train = tensor_test
        x_train = x_train.type(torch.FloatTensor)
        y_train = y_train.type(torch.FloatTensor)

        x_train
        y_train

        net = self

        optimizer = optim.Adam(net.parameters(), lr = .00003)
        EPOCHS = epochs

        step = 0

        #train
        for epoch in range(EPOCHS): 
            random = torch.randperm(x_train.shape[0])
            totalLoss = 0
            count = 0
            for j in range(x_train.shape[0]):
                i = random[j]
                y = torch.tensor([0., 0., 0., 0., 0., 0., 0.])
                index = 0
                for k in range(int(y_train[i])):
                    index+=1
                y[index] = 1.
                output = net(x_train[i])
                loss = F.binary_cross_entropy(output, y)
                totalLoss += loss
                writer.add_scalar("loss/train", loss, step)
                step += 1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                count += 1
                print(count/len(x_train))
            print('avg loss: ', totalLoss/len(x_train))

    def test(self, data : str):

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
        tensor_train
        tensor_test

        print(tensor_test)

        #separate data set into training and testing
        x_test = tensor_train
        y_test = tensor_test
        x_test = x_test.type(torch.FloatTensor)
        y_test = y_test.type(torch.FloatTensor)

        x_test
        y_test

        net = self
        writer = SummaryWriter()

        step = 0
        correct = 0
        label = 0
        aiResult = 0

        aiCount = 0
        actual = 0

        total = 0

        #test
        random = torch.randperm(x_test.shape[0])
        with torch.no_grad():
            for j in range(x_test.shape[0]):

                i = random[j]
                label = y_test[i]

                output = net(x_test[i])
                step += 1


                m = max(output)
                index = 0
                for k in range(7):
                    if output[k] == m:
                        aiResult = k
                        break
                    index+=1

                # print('ai: ', aiResult)
                # print('label: ', label)

                if aiResult == label:
                    correct += 1
                # else:
                #     print(output, ' : ', aiResult, '->', label)
                print(aiResult, '->', label)
                total += 1

                aiCount += aiResult
                actual += label


        print('ai: ', aiCount)
        if actual != 0:
            print('actual: ', actual)
            print('ai/actual: ', aiCount/actual )
        return
        