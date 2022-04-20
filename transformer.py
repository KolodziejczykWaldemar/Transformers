import torch
import torch.nn as nn

from decoder import Decoder
from encoder import Encoder


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x, encoder_keys, encoder_values = self.encoder(x)
        # TODO add while loop for decoder side
        # TODO figure out how to return single token per batch in decoder
        x = self.decoder(x, encoder_keys, encoder_values)
        return x