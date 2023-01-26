"""
nn.Module을 이용해서 Linear Regression을 구현해보자

1. Simple Linear Regression
2. MultiVariable Linear Regression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys

torch.manual_seed(1)

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

x_train2 = torch.FloatTensor([[73, 80, 75],
                              [93, 88, 93],
                              [89, 91, 90],
                              [96, 98, 100],
                              [73, 66, 70]])
y_train2 = torch.FloatTensor([[152], [185], [180], [196], [142]])

input_dim  = 1
output_dim = 1

input_dim2  = 3
output_dim2 = 1

model = nn.Linear(input_dim, output_dim)
optimizer = optim.SGD(model.parameters(), lr = 0.01)

model2 = nn.Linear(input_dim2, output_dim2)
optimizer2 = optim.SGD(model2.parameters(), lr = 0.00001)

Epochs = 2000
for epoch in range(Epochs):

    hypothesis  = model(x_train)
    hypothesis2 = model2(x_train2)
    cost = F.mse_loss(hypothesis, y_train)
    cost2 = F.mse_loss(hypothesis2, y_train2)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    optimizer2.zero_grad()
    cost2.backward()
    optimizer2.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} | Cost : {:.6f}'.format(epoch, Epochs, cost.item()))
        print('Epoch {:4d}/{} | Cost : {:.6f}'.format(epoch, Epochs, cost2.item()))

