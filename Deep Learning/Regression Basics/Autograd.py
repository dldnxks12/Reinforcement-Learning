"""
    Tensor에는 requires_grad 라는 인자가 있다.
    이를 True로 설정하면 해당 변수에 자동으로 미분 기능이 적용된다.
    즉, 미분해서 기울기를 구하고자 하는 변수들에 requires_grad = True로 설정하고,
    해당 변수들이 들어있는 수식에 대해 backward를 수행하면 미분이 계산된다.
"""

import torch

w = torch.tensor(2.0, requires_grad=True)

y = (w)**2
z = (y * 2) + 5

z.backward()

print(w.grad)