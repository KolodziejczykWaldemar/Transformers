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
                 enc_sos_token_id: int,
                 enc_pad_token_id: int,
                 dec_sos_token_id: int,
                 dec_pad_token_id: int,
                 dec_eos_token_id: int,
                 enc_vocab_size: int,
                 dec_vocab_size: int,
                 embedding_dim: int = 512,
                 blocks_number: int = 8,
                 key_dim: int = 64,
                 heads_number: int = 8,
                 hidden_dim: int = 2048,
                 layer_norm_gain: int = 1):
        super().__init__()
        self.enc_sos_token_id = enc_sos_token_id
        self.enc_pad_token_id = enc_pad_token_id
        self.dec_sos_token_id = dec_sos_token_id
        self.dec_pad_token_id = dec_pad_token_id
        self.dec_eos_token_id = dec_eos_token_id

        self.enc_embeddings = TransformerEmbedding(enc_vocab_size, embedding_dim)
        self.encoder = Encoder(key_dim=key_dim,
                               embedding_dim=embedding_dim,
                               heads_number=heads_number,
                               hidden_dim=hidden_dim,
                               blocks_number=blocks_number,
                               layer_norm_gain=layer_norm_gain)

        self.dec_embeddings = TransformerEmbedding(dec_vocab_size, embedding_dim)
        self.decoder = Decoder(vocabulary_size=dec_vocab_size,
                               blocks_number=blocks_number,
                               key_dim=key_dim,
                               embedding_dim=embedding_dim,
                               heads_number=heads_number,
                               hidden_dim=hidden_dim,
                               layer_norm_gain=layer_norm_gain)

    def forward(self,
                encoder_inputs: torch.Tensor,
                decoder_inputs: torch.Tensor) -> torch.Tensor:

        enc_padding_mask = self._get_padding_mask(encoder_inputs, self.enc_pad_token_id)
        dec_padding_mask = self._get_padding_mask(decoder_inputs, self.dec_pad_token_id)

        enc_inputs_emb = self.enc_embeddings(encoder_inputs)
        dec_inputs_emb = self.dec_embeddings(decoder_inputs)

        encoder_outputs = self.encoder(enc_inputs_emb,
                                       enc_padding_mask)
        decoded_output = self.decoder(dec_inputs_emb,
                                      encoder_outputs,
                                      enc_padding_mask,
                                      dec_padding_mask)
        return decoded_output

    def predict_autoregressive(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded_inputs = self.encoder(inputs)

        # TODO change the list below to batched tensor
        decoded_output = [self.sos_token_id]
        current_decoded_token_id = None

        # TODO change this condition to many sequences in batch - maybe masking...?
        while current_decoded_token_id is not self.eos_token_id:
            current_decoded_token_id = self.decoder(decoded_output, encoded_inputs)[-1]
            decoded_output.append(current_decoded_token_id)
        return decoded_output

    def _get_padding_mask(self, documents: torch.Tensor, pad_token_id: int) -> torch.Tensor:
        """
        documents.shape = (batch_size, tokens_in_documents)
        """
        return documents == pad_token_id
