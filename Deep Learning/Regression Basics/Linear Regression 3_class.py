import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class LinearRegressionClass(nn.Module): # nn.Module을 상속
    def __init__(self, input_dim, output_dim):  # Class instance가 생성될 때 세팅되는 항목들
        super.__init__()  # nn.Module의 init에 있던 것들이 세팅된다.
        self.Linear = nn.Linear(input_dim, output_dim)

    def forward(self, x): # 객체를 사용하면 자동으로 수행된다.
        return self.Linear(x)


# Simple Linear Regression model
linear_model = LinearRegressionClass(input_dim = 1, output_dim = 1)

# 데이터
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

optimizer = optim.SGD(linear_model.parameters(), lr = 0.001)
Epochs = 2000

for epoch in range(Epochs):
    hypothesis = linear_model(x_train)
    cost = F.mse_loss(hypothesis, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()