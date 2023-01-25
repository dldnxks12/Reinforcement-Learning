import torch

w = torch.tensor(2.0, requires_grad = True)

Epochs = 10
for epoch in range(Epochs):

    z = 2*w
    z.backward()
    """
    pytorch는 미분을 통해 얻은 기울기를 이전 값에 누적시키는 특징이 있다.
    따라서 optimizer.zero_grad()가 필요!
    """

    print("식을 2로 미분한 값 : {}".format(w.grad))


