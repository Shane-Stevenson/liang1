import torch
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import csv

class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(8000, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 1024),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(1024, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 8000),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def train(self, csv_file : str, epochs : int):
        
        # Validation using MSE Loss function
        loss_function = torch.nn.MSELoss()
        
        # Using an Adam Optimizer with lr = 0.1
        optimizer = torch.optim.Adam(self.parameters(),
                                    lr = .0003,
                                    weight_decay = 1e-8)

        file = open(csv_file)
        csvreader = csv.reader(file)
        csv_voxel_grid = list(csvreader)

        count = 0
        for epoch in range(epochs):
            for v in csv_voxel_grid:
            
                
                count += 1
                a = [float(item) for item in v]
                a.pop()
                a = torch.FloatTensor(a)
                
                # Output of Autoencoder
                reconstructed = self(a)

                # Calculating the loss function
                loss = loss_function(reconstructed, a)

                # The gradients are set to zero,
                # the gradient is computed and stored.
                # .step() performs parameter update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            print(loss)

