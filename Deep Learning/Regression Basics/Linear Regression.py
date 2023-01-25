import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Fix random seed
torch.manual_seed(1)

# Define train variable
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

W = torch.zeros(1, requires_grad=True)  # 학습을 통해 값이 변경되는 변수임을 명시
b = torch.zeros(1, requires_grad=True)

# Gradient Descent Optimizer
optimizer = optim.SGD([W, b], lr = 0.01) # 학습 대상인 W,b가 입력

Epochs = 2000

for epoch in range(Epochs):
    # Hypothesis
    hypothesis = x_train*W + b # y = x*W + b

    # Cost function - loss function
    cost = torch.mean( (hypothesis - y_train)**2)

    # Optimizing
    optimizer.zero_grad() # 미분을 통해 얻은 기울기를 0으로 초기화 -> 새로운 가중치 편향에 대해 새로 계산해야한다.
    cost.backward() # 비용 함수를 미분하여 Gradient 계산
    optimizer.step() # W와 b를 업데이트

    if epoch % 100 == 0:
        print('Epoch : {:4d}/{} || W : {:.3f} || b : {:.3f} || Cost : {:.6f}'.format(epoch, Epochs, W.item(), b.item(), cost.item()))






