import torch
import math
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import functional as F, init


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device = None, dtype = None):
        super().__init__()
        self.factory_kwargs = {"device" : device, "dtype" : dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features,  **self.factory_kwargs))
        self.reset_parameter()

    def reset_parameter(self):
        sigma = 2 / (self.in_features + self.out_features)
        std = math.sqrt(sigma)
        init.trunc_normal_(self.weight, mean = 0, std = std, a = -3 * std, b = 3 * std)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T

nn.Embedding

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device = None, dtype = None):
        super().__init__()
        self.factory_kwargs = {"device" : device, "dtype" : dtype}
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim,  **self.factory_kwargs))
        self.reset_parameter()

    def reset_parameter(self):
        sigma = 1
        std = math.sqrt(sigma)
        init.trunc_normal_(self.weight, mean = 0, std = std, a = -3 * std, b = 3 * std)
    

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedding = self.weight[token_ids]
        return embedding





if __name__ == '__main__':
    print("------------------------------Test Linear------------------------------")
    in_features = 3
    out_features = 4
    factory_kwargs = {"device" : "cpu", "dtype" : torch.float32}
    
    linear_layer = Linear(in_features, out_features, **factory_kwargs)
    inputs = torch.randn(3)
    out = linear_layer(inputs)
    print(linear_layer.weight.data)
    print(out.shape)

    print("------------------------------Test Embedding------------------------------")

    num_embeddings = 100
    embedding_dim = 300
    inputs = torch.randint(0, 100, (30, 10))
    print(f"input shape {inputs.shape}")
    embedding_layer = Embedding(num_embeddings, embedding_dim, **factory_kwargs)
    embed = embedding_layer(inputs)
    
    print(f"after embedding shape is : {embed.shape}")
    


