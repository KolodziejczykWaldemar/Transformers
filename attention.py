import torch
import torch.nn as nn


class MultiheadedAttentionIterative(nn.Module):
    def __init__(self,
                 key_dim: int = 64,
                 embedding_dim: int = 512,
                 heads_number: int = 8,
                 masked: bool = False):
        super().__init__()
        self.key_dim = torch.tensor(key_dim)
        self.embedding_dim = embedding_dim
        self.heads_number = heads_number
        self.masked = masked

        self.query_weights = torch.rand(size=(self.heads_number, self.embedding_dim, self.key_dim))
        # TODO maybe (self.embedding_dim, self.key_dim * self.heads_number)
        self.key_weights = torch.rand(size=(self.heads_number, self.embedding_dim, self.key_dim))
        self.value_weights = torch.rand(size=(self.heads_number, self.embedding_dim, self.key_dim))
        self.output_weights = torch.rand(
            size=(self.key_dim * self.heads_number, self.embedding_dim))

    def forward(self, x):
        if len(x.shape) == 2:
            x = torch.unsqueeze(x, dim=0)
        batch_size = x.shape[0]
        tokens_in_document = x.shape[1]

        multiheaded_attention = torch.empty(batch_size, tokens_in_document,
                                            self.key_dim * self.heads_number)
        for head_id in range(self.heads_number):
            queries = torch.matmul(x, self.query_weights[head_id, :, :])
            keys = torch.matmul(x, self.key_weights[head_id, :, :])
            values = torch.matmul(x, self.value_weights[head_id, :, :])
            # TODO return keys and values

            attention_head = torch.matmul(queries, torch.transpose(keys, dim0=-2, dim1=-1))
            attention_head = torch.divide(attention_head, torch.sqrt(self.key_dim))
            if self.masked:
                mask = torch.ones((tokens_in_document, tokens_in_document), dtype=torch.bool)
                mask = torch.triu(mask, diagonal=1)
                attention_head = attention_head.masked_fill(mask, -torch.inf)
            attention_head = torch.softmax(attention_head, dim=-1)

            attention_head = torch.matmul(attention_head, values)
            multiheaded_attention[:, :,
            head_id * self.key_dim: (head_id + 1) * self.key_dim] = attention_head
            # TODO vectorize it to avoid dummy looping
        output = torch.matmul(multiheaded_attention, self.output_weights)
        return output


class MultiheadedAttention(nn.Module):
    def __init__(self,
                 key_dim: int = 64,
                 embedding_dim: int = 512,
                 heads_number: int = 8,
                 masked: bool = False):
        super().__init__()
        self.key_dim = torch.tensor(key_dim)
        self.embedding_dim = embedding_dim
        self.heads_number = heads_number
        self.masked = masked

        self.query_weights = torch.rand(size=(self.embedding_dim, self.key_dim * self.heads_number))
        self.output_weights = torch.rand(
            size=(self.key_dim * self.heads_number, self.embedding_dim))

    def forward(self, x, keys, values):
        if len(x.shape) == 2:
            x = torch.unsqueeze(x, dim=0)
        batch_size = x.shape[0]
        tokens_in_document = x.shape[1]

        # shape: batch_size, tokens_in_documents, heads_number * key_dim
        queries = torch.matmul(x, self.query_weights)

        # shape: batch_size, tokens_in_documents, heads_number, key_dim
        queries = queries.view(batch_size, -1, self.heads_number, self.key_dim)

        # shape: batch_size, heads_number, tokens_in_documents, key_dim
        queries = torch.transpose(queries, dim0=1, dim1=2)

        # shape: batch_size, heads_number, tokens_in_documents, tokens_in_documents
        attention_head = torch.matmul(queries, torch.transpose(keys, dim0=-2, dim1=-1))
        attention_head = torch.divide(attention_head, torch.sqrt(self.key_dim))
        if self.masked:
            mask = torch.ones((tokens_in_document, tokens_in_document), dtype=torch.bool)
            mask = torch.triu(mask, diagonal=1)
            attention_head = attention_head.masked_fill(mask, -torch.inf)
        attention_head = torch.softmax(attention_head, dim=-1)

        # shape: batch_size, heads_number, tokens_in_documents, key_dim
        attention_head = torch.matmul(attention_head, values)

        # shape: batch_size, tokens_in_documents, heads_number, key_dim
        attention_head = torch.transpose(attention_head, dim0=1, dim1=2)

        # shape: batch_size, tokens_in_documents, heads_number * key_dim
        attention_head = attention_head.contiguous().view(batch_size, -1, self.heads_number * self.key_dim)

        # shape: batch_size, tokens_in_documents, embedding_dim
        output = torch.matmul(attention_head, self.output_weights)
        return output


class MultiheadedSelfAttention(MultiheadedAttention):
    def __init__(self,
                 key_dim: int = 64,
                 embedding_dim: int = 512,
                 heads_number: int = 8,
                 masked: bool = False):
        super().__init__(key_dim=key_dim,
                         embedding_dim=embedding_dim,
                         heads_number=heads_number,
                         masked=masked)
        self.key_weights = torch.rand(size=(self.embedding_dim, self.key_dim * self.heads_number))
        self.value_weights = torch.rand(size=(self.embedding_dim, self.key_dim * self.heads_number))

    def forward(self, x):
        if len(x.shape) == 2:
            x = torch.unsqueeze(x, dim=0)
        batch_size = x.shape[0]

        keys = torch.matmul(x, self.key_weights)
        values = torch.matmul(x, self.value_weights)

        keys = keys.view(batch_size, -1, self.heads_number, self.key_dim)
        values = values.view(batch_size, -1, self.heads_number, self.key_dim)

        keys = torch.transpose(keys, dim0=1, dim1=2)
        values = torch.transpose(values, dim0=1, dim1=2)

        outputs = super().forward(x, keys, values)
        return outputs, keys, values
