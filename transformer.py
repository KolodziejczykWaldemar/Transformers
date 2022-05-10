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
                 blocks_number: int = 6,
                 key_dim: int = 64,
                 heads_number: int = 8,
                 hidden_dim: int = 2048):
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
                               blocks_number=blocks_number)

        self.dec_embeddings = TransformerEmbedding(dec_vocab_size, embedding_dim)
        self.decoder = Decoder(vocabulary_size=dec_vocab_size,
                               blocks_number=blocks_number,
                               key_dim=key_dim,
                               embedding_dim=embedding_dim,
                               heads_number=heads_number,
                               hidden_dim=hidden_dim)

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
                                      dec_padding_mask,
                                      enc_padding_mask)
        return decoded_output

    def predict_autoregressive(self, inputs: torch.Tensor, max_len: int = 500) -> torch.Tensor:
        batch_size = inputs.shape[0]

        enc_padding_mask = self._get_padding_mask(inputs, self.enc_pad_token_id)
        enc_inputs_emb = self.enc_embeddings(inputs)

        encoder_outputs = self.encoder(enc_inputs_emb,
                                       enc_padding_mask)

        decoder_inputs = torch.ones(size=(batch_size, 1), dtype=torch.long) * self.dec_sos_token_id

        ended_samples = torch.zeros(batch_size).bool()
        for _ in range(max_len):

            dec_inputs_emb = self.dec_embeddings(decoder_inputs)
            dec_padding_mask = self._get_padding_mask(decoder_inputs, self.dec_pad_token_id)

            current_decoded = self.decoder(dec_inputs_emb,
                                           encoder_outputs,
                                           dec_padding_mask,
                                           enc_padding_mask)
            current_decoded_token_ids = torch.argmax(current_decoded[:, -1, :], dim=-1)
            decoder_inputs = torch.cat((decoder_inputs, current_decoded_token_ids.unsqueeze(1)), dim=1)

            ended_samples[current_decoded_token_ids == self.dec_eos_token_id] = True

            if ended_samples.all():
                break

        return decoder_inputs

    def _get_padding_mask(self, documents: torch.Tensor, pad_token_id: int) -> torch.Tensor:
        """
        documents.shape = (batch_size, tokens_in_documents)
        """
        return documents == pad_token_id
