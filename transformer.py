from enum import Enum

import torch
import torch.nn as nn

from decoder import Decoder
from embedding import TransformerEmbedding
from encoder import Encoder


class SpecialToken(Enum):
    PAD_WORD = '<BLK>'
    UNK_WORD = '<UNK>'
    SOS_WORD = '<S>'
    EOS_WORD = '</S>'


class Transformer(nn.Module):
    def __init__(self,
                 sos_token_id: int,
                 eos_token_id: int,
                 pad_token_id: int,
                 vocabulary_size: int,
                 blocks_number: int = 8,
                 key_dim: int = 64,
                 embedding_dim: int = 512,
                 heads_number: int = 8,
                 hidden_dim: int = 2048,
                 layer_norm_gain: int = 1):
        super().__init__()
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.embeddings = TransformerEmbedding(vocabulary_size, embedding_dim)
        self.encoder = Encoder(key_dim=key_dim,
                               embedding_dim=embedding_dim,
                               heads_number=heads_number,
                               hidden_dim=hidden_dim,
                               blocks_number=blocks_number,
                               layer_norm_gain=layer_norm_gain)
        self.decoder = Decoder(vocabulary_size=vocabulary_size,
                               blocks_number=blocks_number,
                               key_dim=key_dim,
                               embedding_dim=embedding_dim,
                               heads_number=heads_number,
                               hidden_dim=hidden_dim,
                               layer_norm_gain=layer_norm_gain)

    def forward(self, encoder_inputs, decoder_inputs):

        encoder_padding_mask = self._get_padding_mask(encoder_inputs)
        decoder_padding_mask = self._get_padding_mask(decoder_inputs)

        encoder_inputs_emb = self.embeddings(encoder_inputs)
        decoder_inputs_emb = self.embeddings(decoder_inputs)

        encoder_outputs = self.encoder(encoder_inputs_emb,
                                       encoder_padding_mask)
        decoded_output = self.decoder(decoder_inputs_emb,
                                      encoder_outputs,
                                      encoder_padding_mask,
                                      decoder_padding_mask)
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

    def _get_padding_mask(self, documents):
        """
        documents.shape = (batch_size, tokens_in_documents)
        """
        return documents == self.pad_token_id
