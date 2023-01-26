import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

x_train  =  torch.FloatTensor([[73,  80,  75],
                               [93,  88,  93],
                               [89,  91,  90],
                               [96,  98,  100],
                               [73,  66,  70]])
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])

# dataset을 만든다 -> 이걸로 dataloader 사용 가능
dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=2, shuffle = True)

model = nn.Linear(3,1)
optimizer = optim.SGD(model.parameters(), lr = 0.00001)

Epochs = 20
for epoch in range(Epochs + 1):
    for batch_idx, samples in enumerate(dataloader):
        x_train = samples[0]
        y_train = samples[1]

        hypothesis = model(x_train)
        cost = F.mse_loss(hypothesis, y_train)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
        epoch, Epochs, batch_idx+1, len(dataloader),
        cost.item()
        ))
