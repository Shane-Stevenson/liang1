import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

"""
Final iteration of our regression AI
"""

#Not heavily documented because this mode is very primitive and not too useful

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(30*30*20, 4096) #First layer must be equal to the amount of voxels in a filled grid
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 128)
        self.fc7 = nn.Linear(128, 7)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)

        return F.softmax(x, dim = -1)
    
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

                n = np.zeros((output.size()))
                adjLabels = torch.FloatTensor(n).to('mps')
                for i,val in enumerate(labels):
                    adjLabels[i][int(val)] = 1

                batchLoss = F.binary_cross_entropy(output, adjLabels)
                optimizer.zero_grad()
                batchLoss.backward()
                optimizer.step()
                totalLoss += batchLoss.item()
                count+=1

                print(count/l)
            
            print('avg loss: ', totalLoss/count)

    def test(self, DataLoader : torch.utils.data.DataLoader):
        aiCount = 0
        treeCount = 0
        with torch.no_grad():
            for batch in DataLoader:
                data, labels = batch
                output = self(data.to('mps'))

                aiResult = []
                for i in output:
                    for j, val in enumerate(i):
                        if val == max(i):
                            aiResult.append(j)
                
                l1 = []

                for i in range(len(aiResult)):
                    aiCount += aiResult[i]
                    l1.append(aiResult[i])

                l2 = []
                for i in labels:
                    treeCount += i.item()
                    l2.append(i.item())

                for i in range(len(l1)):
                    print(l1[i], '->', l2[i])
            

        print('ai: ', aiCount)
        print('treeCount: ', treeCount)
        print('accuracy: ', aiCount/treeCount)