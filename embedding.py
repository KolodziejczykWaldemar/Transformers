import numpy as np
import torch.nn as nn
import torch


class TransformerEmbedding(nn.Embedding):
    def __init__(self,
                 vocabulary_size: int,
                 embedding_dim: int = 512):
        super().__init__(vocabulary_size, embedding_dim)
        self.drop_out = nn.Dropout(p=0.1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if len(input.shape) == 1:
            tokens_in_document = input.shape[0]
        elif len(input.shape) == 2:
            tokens_in_document = input.shape[1]
        else:
            raise Exception(f'Input should be passed with 1 or 2 dimensions, got {input.shape}')

        semantic_embeddings = super().forward(input)
        positional_embeddings = self._get_positional_encoding(tokens_in_document=tokens_in_document)
        embeddings = semantic_embeddings + positional_embeddings
        embeddings = self.drop_out(embeddings)
        return embeddings

    def _get_angles(self,
                    token_positions: np.ndarray,
                    embedding_dims: np.ndarray) -> np.ndarray:
        angle_rates = 1 / np.power(10000,
                                   (2 * (embedding_dims // 2)) / np.float32(self.embedding_dim))
        return token_positions * angle_rates

    def _get_positional_encoding(self, tokens_in_document: int) -> torch.Tensor:

        angle_rads = self._get_angles(np.arange(tokens_in_document)[:, np.newaxis],
                                      np.arange(self.embedding_dim)[np.newaxis, :])

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return nn.Parameter(torch.tensor(pos_encoding, dtype=torch.float32), requires_grad=False)
