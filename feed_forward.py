import torch
import torch.nn as nn


class PositionWiseDenseNetwork(nn.Module):
    def __init__(self, hidden_dim: int = 2048, embedding_dim: int = 512, dropout_prob: float = 0.1):
        super(PositionWiseDenseNetwork, self).__init__()
        self.inner_linear = nn.Linear(embedding_dim, hidden_dim, bias=True)
        self.outer_linear = nn.Linear(hidden_dim, embedding_dim, bias=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = self.inner_linear(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.outer_linear(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, embedding_dim: int = 512, epsilon: float = 1e-12):
        """Based on: "Layer Normalization" by Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton

        https://arxiv.org/pdf/1607.06450.pdf
        """
        super(LayerNorm, self).__init__()
        self.gain = nn.Parameter(torch.ones(embedding_dim))
        self.bias = nn.Parameter(torch.zeros(embedding_dim))
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        out = (x - mean) / (std + self.epsilon)
        out = self.gain * out + self.bias
        return out
