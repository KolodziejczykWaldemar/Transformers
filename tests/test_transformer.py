from unittest import TestCase

import numpy as np
import torch
import torch.nn as nn

from attention import MultiheadedSelfAttention
from decoder import Decoder, DecoderBlock
from embedding import TransformerEmbedding
from transformer import Transformer


class TestTransformer(TestCase):
    def setUp(self) -> None:
        embedding_dim = 20
        vocabulary_size = 45
        self.model = Transformer(enc_vocab_size=vocabulary_size,
                                 dec_vocab_size=vocabulary_size,
                                 embedding_dim=embedding_dim,
                                 enc_sos_token_id=1,
                                 enc_pad_token_id=0,
                                 dec_sos_token_id=1,
                                 dec_eos_token_id=2,
                                 dec_pad_token_id=0
                                 )

    def test_dimensionality_many_documents(self) -> None:
        encoder_input = torch.tensor(np.array([[0, 2, 3, 4, 3],
                                               [2, 1, 4, 1, 6]]))
        decoder_input = torch.tensor(np.array([[1, 2, 3, 4, 0, 0],
                                               [7, 6, 9, 3, 2, 1]]))

        output = self.model(encoder_input, decoder_input)


class TestDecoder(TestCase):
    def setUp(self) -> None:
        self.embedding_dim = 20
        vocabulary_size = 45
        self.decoder = Decoder(vocabulary_size=vocabulary_size,
                               embedding_dim=self.embedding_dim)

    def test_decoder_block(self):
        decoder_block = DecoderBlock(embedding_dim=self.embedding_dim)
        encoder_output = torch.rand(size=(2, 8, self.embedding_dim))
        decoder_input = torch.rand(size=(2, 6, self.embedding_dim))
        encoder_padding_mask = torch.tensor(np.array([[True, True, True, False, False, False, False, False],
                                                      [True, True, True, False, False, False, False, False]]))
        decoder_padding_mask = torch.tensor(np.array([[False, False, False, False, True, True],
                                                      [False, False, False, False, True, True]]))

        output = decoder_block(decoder_input, encoder_output, encoder_padding_mask, decoder_padding_mask)
        print()

    def test_dimensionality(self) -> None:
        encoder_output = torch.rand(size=(2, 8, self.embedding_dim))
        decoder_input = torch.rand(size=(2, 6, self.embedding_dim))
        encoder_padding_mask = torch.tensor(np.array([[True, True, True, False, False, False, False, False],
                                                      [True, True, True, False, False, False, False, False]]))
        decoder_padding_mask = torch.tensor(np.array([[False, False, False, False, True, True],
                                                      [False, False, False, False, True, True]]))

        output = self.decoder(decoder_input, encoder_output, encoder_padding_mask, decoder_padding_mask)
