import torch
import torch.nn as nn


class PositionWiseDenseNetwork(nn.Module):
    def __init__(self, hidden_dim: int = 2048, embedding_dim: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.inner_weights = torch.rand(size=(self.embedding_dim, self.hidden_dim))
        self.outer_weights = torch.rand(size=(self.hidden_dim, self.embedding_dim))
        self.inner_biases = torch.rand(size=(self.hidden_dim,))
        self.outer_biases = torch.rand(size=(self.embedding_dim,))

        nn.init.xavier_uniform_(self.inner_weights)
        nn.init.xavier_uniform_(self.outer_weights)

    def forward(self, x):
        x = torch.matmul(x, self.inner_weights)
        x = x + self.inner_biases
        x = torch.relu(x)
        x = torch.matmul(x, self.outer_weights)
        x = x + self.outer_biases
        return x


class LayerNorm(nn.Module):
    """Based on: "Layer Normalization" by Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton

    https://arxiv.org/pdf/1607.06450.pdf
    """
    def __init__(self, gain: int = 1):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        means = torch.unsqueeze(torch.mean(x, -1), 2)
        stds = torch.unsqueeze(torch.std(x, -1), 2)
        x = (x - means) * self.gain / stds
        return x

