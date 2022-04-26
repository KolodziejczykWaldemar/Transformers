import torch
import torch.nn as nn

from decoder import Decoder
from encoder import Encoder


class Transformer(nn.Module):
    def __init__(self, sos_token_id: int, eos_token_id: int):
        super().__init__()
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, inputs):
        encoded_inputs = self.encoder(inputs)
        decoded_output = self.decoder(inputs, encoded_inputs)
        return decoded_output

    def predict_autoregressive(self, inputs):
        encoded_inputs = self.encoder(inputs)

        # TODO change the list below to batched tensor
        decoded_output = [self.sos_token_id]
        current_decoded_token_id = None

        # TODO change this condition to many sequences in batch - maybe masking...?
        while current_decoded_token_id is not self.eos_token_id:
            current_decoded_token_id = self.decoder(decoded_output, encoded_inputs)[-1]
            decoded_output.append(current_decoded_token_id)
        return decoded_output
