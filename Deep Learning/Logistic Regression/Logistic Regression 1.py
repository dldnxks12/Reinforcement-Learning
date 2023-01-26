import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


torch.manual_seed(1)

x_train = torch.FloatTensor([[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]) # 6 x 2
y_train = torch.FloatTensor([[0], [0], [0], [1], [1], [1]]) # 6 x 1

W = torch.zeros((2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([W,b], lr = 0.1)

Epochs = 1000

for epoch in range(Epochs):

    hypothesis = torch.sigmoid(x_train.matmul(W) + b)
    # or manually hypothesis = 1 / (1 + torch.exp(-(x_train.matmul(W) + b)))

    cost = F.binary_cross_entropy(hypothesis, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, Epochs, cost.item()
        ))


