import numpy as np
import torch.nn as nn
import torch


class TransformerEmbedding(nn.Embedding):
    def __init__(self,
                 vocabulary_size: int,
                 embedding_dim: int = 512,
                 max_doc_len: int = 256,
                 dropout_prob: float = 0.1):
        super().__init__(vocabulary_size, embedding_dim)
        self.positional_embedding = PositionalEmbedding(embedding_dim=embedding_dim,
                                                        max_doc_len=max_doc_len)
        self.drop_out = nn.Dropout(p=dropout_prob)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        semantic_embeddings = super().forward(input)
        positional_embeddings = self.positional_embedding(input)
        embeddings = semantic_embeddings + positional_embeddings
        embeddings = self.drop_out(embeddings)
        return embeddings


class PositionalEmbedding(nn.Module):
    def __init__(self,
                 embedding_dim: int = 512,
                 max_doc_len: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_doc_len = max_doc_len
        self.positional_encoding = self._setup_positional_encoding()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if len(input.shape) == 1:
            tokens_in_document = input.shape[0]
        elif len(input.shape) == 2:
            tokens_in_document = input.shape[1]
        else:
            raise Exception(f'Input should be passed with 1 or 2 dimensions, got {input.shape}')

        positional_embeddings = self.positional_encoding[:, :tokens_in_document, :]
        return positional_embeddings

    def _get_angles(self,
                    token_positions: np.ndarray,
                    embedding_dims: np.ndarray) -> np.ndarray:
        angle_rates = 1 / np.power(10000,
                                   (2 * (embedding_dims // 2)) / np.float32(self.embedding_dim))
        return token_positions * angle_rates

    def _setup_positional_encoding(self) -> torch.Tensor:

        angle_rads = self._get_angles(np.arange(self.max_doc_len)[:, np.newaxis],
                                      np.arange(self.embedding_dim)[np.newaxis, :])

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return nn.Parameter(torch.tensor(pos_encoding, dtype=torch.float32), requires_grad=False)
