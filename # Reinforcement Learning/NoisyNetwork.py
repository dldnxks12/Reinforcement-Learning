"""

*reference : https://github.com/thomashirtz/noisy-networks


Noisy Network를 바로 적용하기 편하게 Module로 만들어놓자.

"""

import torch
import torch.nn as nn

class AbstractNoisyLayer(nn.Module):
    # : , ->       그냥 주석의 역할
    def __init__(self, input_features : int, output_features : int, sigma : int):
        super(nn.Module).__init__()

        self.sigma           = sigma
        self.input_features  = input_features
        self.output_features = output_features

        """
        nn.Parameter -> nn.Module 클래스의 attribute로 할당하면 자동으로 학습 파라미터 list에 추가된다.
        """
        self.mu_bias      = nn.Parameter(torch.FloatTensor(output_features))                 # size p
        self.mu_weight    = nn.Parameter(torch.FloatTensor(output_features, input_features)) # size p x q

        self.sigma_bias   = nn.Parameter(torch.FloatTensor(output_features))
        self.sigma_weight = nn.Parameter(torch.FloatTensor(output_features, input_features))

        """
        self.register_buffer로 layer를 등록하면?
        
            - optimizer가 업데이트 하지 않는다. 
            - 즉, End-to-end로 네트워크를 구성할 때 중간에 업데이트 하지 않는 일단 layer를 넣을 때 사용할 수 있다.  
            - GPU에서 작동
            - state_dict에는 저장이 된다. (값이 확인된다.)             
        """
        self.register_buffer('epsilon_input', torch.FloatTensor(input_features))
        self.register_buffer('epsilon_output', torch.FloatTensor(output_features))


    def forward(self,
                x : torch.Tensor,
                sample_noise : bool = True
                ):
        if not self.training:
            return nn.functional.linear(x, weight = self.mu_weight, bias = self.mu_bias)

        if sample_noise:
            self.sample_noise()

        return nn.functional.linear(x, weight=self.weight, bias = self.bias)

    @property
    def weight(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def bias(self)  -> torch.tensor:
        raise NotImplementedError


class IndependentNoisyLayer(AbstractNoisyLayer):
    def __init__(self):
        pass