import torch
import torch.nn as nn

from decoder import Decoder
from encoder import Encoder


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, inputs):
        encoded_inputs = self.encoder(inputs)
        # TODO add while loop for decoder side
        # TODO figure out how to return single token per batch in decoder
        decoded_output = self.decoder(inputs, encoded_inputs)
        return decoded_output