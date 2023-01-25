
# - Make Tensor with Numpy - #
import numpy as np

# 1-D Array
t = np.array([0, 1, 2, 3, 4, 5])
print("Rank  :", t.ndim) # 몇 차원인가 ? 1차원 : 벡터 / 2차원 : 행렬 / 3차원 : 텐서
print("Shape :", t.shape) # (6, ) = (1,6)을 의미

# 2-D Array
t = np.array([ [1, 2, 3], [4, 5, 6], [ 7, 8, 9], [10, 11, 12]])

print("Rank  :", t.ndim) # 몇 차원인가? 1차원 : 벡터 / 2차원 : 행렬 / 3차원 : 텐서
print("Shape :", t.shape) # (4,3)


# - Make Tensor with Pytorch - #
import torch

# 1-D Tensor
t = torch.FloatTensor([0, 1, 2, 3, 4, 5, 6])

print("Rank  :", t.dim())  # 몇 차원?
print("Shape :", t.shape) # Tensor의 Shape
print("Size  :", t.size()) # Tensor의 Shape

# 2-D Tensor
t = torch.FloatTensor([[1,2,3],
                       [4,5,6],
                       [7,8,9],
                       [10,11,12]])

print("Rank  :", t.dim())  # 몇 차원?
print("Shape :", t.shape) # Tensor의 Shape
print("Size  :", t.size()) # Tensor의 Shape

