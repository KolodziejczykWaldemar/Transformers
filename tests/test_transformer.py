from unittest import TestCase

import numpy as np
import torch
import torch.nn as nn

from attention import MultiheadedSelfAttention
from embedding import TransformerEmbedding
from transformer import Transformer


class TestTransformer(TestCase):
    def setUp(self) -> None:
        embedding_dim = 20
        vocabulary_size = 45
        self.model = Transformer(vocabulary_size=vocabulary_size,
                                 embedding_dim=embedding_dim,
                                 sos_token_id=1,
                                 eos_token_id=2,
                                 pad_token_id=0)

    def test_dimensionality_many_documents(self) -> None:
        encoder_input = torch.tensor(np.array([[0, 2, 3, 4, 3],
                                               [2, 1, 4, 1, 6]]))
        decoder_input = torch.tensor(np.array([[1, 2, 3, 4, 0, 0],
                                               [7, 6, 9, 3, 2, 1]]))

        output = self.model(encoder_input, decoder_input)
        self.assertEqual(decoder_input.shape, output.shape)