import torch
import math
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import functional as F, init

nn.Linear

class Linear(nn.Module):
    """
    y = wx + b
    """
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, 
                in_features, 
                out_features, 
                bias=True, 
                device=None, 
                dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        print(factory_kwargs)
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.empty(out_features, in_features, **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameter()

    def reset_parameter(self) -> None:
        init.kaiming_uniform_(self.weight, math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
        

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

if __name__ == '__main__':
    l1 = Linear(3, 4, False, "cpu", torch.float32)
    in_feature = torch.randn(3)
    out_feature = l1(in_feature)
    print(l1.weight.data)
    print(out_feature.shape)