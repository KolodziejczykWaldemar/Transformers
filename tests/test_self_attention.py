from unittest import TestCase

import numpy as np
import torch
import torch.nn as nn

from attention import MultiheadedSelfAttention
from embedding import TransformerEmbedding


class TestMultiheadedAttention(TestCase):
    def setUp(self) -> None:
        embedding_dim = 20
        vocabulary_size = 5
        self.embeddings = TransformerEmbedding(vocabulary_size, embedding_dim)
        self.model = MultiheadedSelfAttention(embedding_dim=embedding_dim)

    def test_dimensionality_many_documents(self) -> None:
        tokens = torch.tensor(np.array([[0, 2, 3],
                                        [2, 1, 4]]))
        embeddings = self.embeddings(tokens)

        attention_values = self.model(embeddings)
        self.assertEqual(embeddings.shape, attention_values.shape)

    def test_repetitive_results(self) -> None:
        document_ids = [0, 2, 3]
        tokens = torch.tensor(np.array([document_ids, document_ids]))
        embeddings = self.embeddings(tokens)

        attention_values = self.model(embeddings)
        attention_values = attention_values.detach().numpy()
        attention_values_first_document = attention_values[0, :, :]
        attention_values_second_document = attention_values[1, :, :]
        self.assertTrue(np.array_equal(attention_values_first_document,
                                       attention_values_second_document))

    def test_dimensionality_single_document(self) -> None:
        tokens = torch.tensor(np.array([0, 2, 3]))
        embeddings = self.embeddings(tokens)

        attention_values = self.model(embeddings)
        self.assertEqual(embeddings.shape, attention_values.shape)

    def test_masked(self):
        tokens = torch.tensor(np.array([[0, 2, 3],
                                        [2, 1, 4]]))

        embeddings = self.embeddings(tokens)

        self.model.masked = True
        attention_values = self.model(embeddings)
        # TODO add test

        # dnn = PositionWiseDenseNetwork(embedding_dim=20).forward(attention_values)
        #
        # output = LayerNorm().forward(dnn)
        # print(embeds)
