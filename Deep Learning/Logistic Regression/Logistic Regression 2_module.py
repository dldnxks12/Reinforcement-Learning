import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
x_train = torch.FloatTensor([[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]) # 6 x 2
y_train = torch.FloatTensor([[0], [0], [0], [1], [1], [1]]) # 6 x 1

model = nn.Sequential(
        nn.Linear(2, 1),
        nn.Sigmoid()
)

optimizer = optim.SGD(model.parameters(), lr = 0.1)

Epochs = 2000
for epoch in range(Epochs):
    hypothesis = model(x_train)
    cost = F.binary_cross_entropy(hypothesis, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

print(model(torch.FloatTensor([1, 2]))) # OK 
