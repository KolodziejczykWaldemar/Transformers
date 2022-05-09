from typing import Optional

import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                queries: torch.Tensor,
                keys: torch.Tensor,
                values: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        queries.shape = (batch_size, tokens_in_documents, heads_number, key_dim)
        mask.shape = (batch_size, heads_number, queries_number, keys_number)
        """
        key_dim = nn.Parameter(torch.tensor(keys.shape[-1]), requires_grad=False)

        # shape: batch_size, heads_number, tokens_in_documents, key_dim
        queries = queries.transpose(dim0=1, dim1=2)
        keys = keys.transpose(dim0=1, dim1=2)
        values = values.transpose(dim0=1, dim1=2)

        # shape: batch_size, heads_number, tokens_in_documents, tokens_in_documents
        attention = torch.matmul(queries, torch.transpose(keys, dim0=-2, dim1=-1))

        attention = torch.divide(attention, torch.sqrt(key_dim))
        if mask is not None:
            attention = attention.masked_fill(mask, -1e10)
        attention = torch.softmax(attention, dim=-1)

        # shape: batch_size, heads_number, tokens_in_documents, key_dim
        output = torch.matmul(attention, values)

        # shape: batch_size, tokens_in_documents, heads_number, key_dim
        return torch.transpose(output, dim0=1, dim1=2)


class MultiheadedAttention(nn.Module):
    def __init__(self,
                 key_dim: int = 64,
                 embedding_dim: int = 512,
                 heads_number: int = 8):
        super().__init__()
        self.key_dim = torch.tensor(key_dim)
        self.embedding_dim = embedding_dim
        self.heads_number = heads_number

        self.query_projection = nn.Linear(self.embedding_dim, self.key_dim * self.heads_number, bias=False)
        self.key_projection = nn.Linear(self.embedding_dim, self.key_dim * self.heads_number, bias=False)
        self.value_projection = nn.Linear(self.embedding_dim, self.key_dim * self.heads_number, bias=False)

        self.output_projection = nn.Linear(self.key_dim * self.heads_number, self.embedding_dim, bias=False)

        self.attention = ScaledDotProductAttention()

    def forward(self,
                queries: torch.Tensor,
                keys: torch.Tensor,
                values: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        queries.shape = (batch_size, tokens_in_documents, embedding_dim)
        keys.shape = values.shape = (batch_size, tokens_in_documents, heads_number, key_dim)
        mask.shape = (batch_size, heads_number, queries_number, keys_number)
        """
        batch_size = queries.shape[0]

        # shape: batch_size, tokens_in_documents, heads_number * key_dim
        queries = self.query_projection(queries)
        keys = self.key_projection(keys)
        values = self.value_projection(values)

        # shape: batch_size, tokens_in_documents, heads_number, key_dim
        queries = queries.view(batch_size, -1, self.heads_number, self.key_dim)
        keys = keys.view(batch_size, -1, self.heads_number, self.key_dim)
        values = values.view(batch_size, -1, self.heads_number, self.key_dim)

        # shape: batch_size, tokens_in_documents, heads_number, key_dim
        attention_heads = self.attention(queries, keys, values, mask=mask)

        # shape: batch_size, tokens_in_documents, heads_number * key_dim
        attention_heads = attention_heads.contiguous().view(batch_size, -1, self.heads_number * self.key_dim)

        # shape: batch_size, tokens_in_documents, embedding_dim
        output = self.output_projection(attention_heads)
        return output
