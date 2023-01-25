import torch
import numpy as np

# view , reshape, squeeze, unsqueeze, stack, concatenate

t = np.array([[[0, 1, 2],
               [3, 4, 5]],
              [[6, 7, 8],
               [9, 10, 11]]])

ft = torch.FloatTensor(t)

#print(ft)
#print(ft.dim()) # 3D

print(ft)
print(ft.shape) # 2 x 2 x 3

print(ft.view([-1,3]))
print(ft.view([-1,3]).shape)
print(ft.shape) # 2 x 2 x 3

