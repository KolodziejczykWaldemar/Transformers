import torch
import torch.nn as nn


class MultiheadedAttentionOldIterative(nn.Module):
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


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, queries, keys, values, mask = None):
        """
        queries.shape = (batch_size, tokens_in_documents, heads_number, key_dim)
        mask.shape = (batch_size, queries_number, keys_number)
        """
        key_dim = keys.shape[-1]

        # shape: batch_size, heads_number, tokens_in_documents, key_dim
        queries = queries.transpose(dim0=1, dim1=2)
        keys = keys.transpose(dim0=1, dim1=2)
        values = values.transpose(dim0=1, dim1=2)

        # shape: batch_size, heads_number, tokens_in_documents, tokens_in_documents
        attention = torch.matmul(queries, torch.transpose(keys, dim0=-2, dim1=-1))

        attention = torch.divide(attention, torch.sqrt(key_dim))
        if mask is not None:
            attention = attention.masked_fill(mask, -torch.inf)
        attention = torch.softmax(attention, dim=-1)

        # shape: batch_size, heads_number, tokens_in_documents, key_dim
        output = torch.matmul(attention, values)

        # shape: batch_size, tokens_in_documents, heads_number, key_dim
        return torch.transpose(output, dim0=1, dim1=2)


class MultiheadedAttentionV2(nn.Module):
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

        # TODO how to initialize weights
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


class MultiheadedAttention(nn.Module):
    def __init__(self,
                 key_dim: int = 64,
                 embedding_dim: int = 512,
                 heads_number: int = 8):
        super().__init__()
        self.key_dim = torch.tensor(key_dim)
        self.embedding_dim = embedding_dim
        self.heads_number = heads_number

        # TODO how to initialize weights
        self.query_weights = torch.rand(size=(self.embedding_dim, self.key_dim * self.heads_number))
        self.output_weights = torch.rand(
            size=(self.key_dim * self.heads_number, self.embedding_dim))
        self.attention = ScaledDotProductAttention()

    def forward(self, inputs, keys, values, mask):
        """
        inputs.shape = (batch_size, tokens_in_documents, embedding_dim)
        keys.shape = values.shape = (batch_size, tokens_in_documents, heads_number, key_dim)
        mask.shape = (batch_size, queries_number, keys_number)
        """
        if len(inputs.shape) == 2:
            inputs = torch.unsqueeze(inputs, dim=0)
        batch_size = inputs.shape[0]
        queries_number = inputs.shape[1]
        keys_number = keys.shape[1]

        # shape: batch_size, tokens_in_documents, heads_number * key_dim
        queries = torch.matmul(inputs, self.query_weights)

        # shape: batch_size, tokens_in_documents, heads_number, key_dim
        queries = queries.view(batch_size, -1, self.heads_number, self.key_dim)

        # shape: batch_size, tokens_in_documents, heads_number, key_dim
        attention_heads = self.attention(queries, keys, values, mask=mask)

        # shape: batch_size, tokens_in_documents, heads_number * key_dim
        attention_heads = attention_heads.contiguous().view(batch_size, -1, self.heads_number * self.key_dim)

        # shape: batch_size, tokens_in_documents, embedding_dim
        output = torch.matmul(attention_heads, self.output_weights)
        return output


class MultiheadedSelfAttention(MultiheadedAttention):
    def __init__(self,
                 key_dim: int = 64,
                 embedding_dim: int = 512,
                 heads_number: int = 8,
                 is_decoder_layer: bool = False):
        super().__init__(key_dim=key_dim,
                         embedding_dim=embedding_dim,
                         heads_number=heads_number)
        self.is_decoder_layer = is_decoder_layer
        self.key_weights = torch.rand(size=(self.embedding_dim, self.key_dim * self.heads_number))
        self.value_weights = torch.rand(size=(self.embedding_dim, self.key_dim * self.heads_number))

    def forward(self, inputs, padding_mask):
        """
        inputs.shape = (?batch_size, tokens_in_documents, embedding_dim)
        padding_mask.shape = (batch_size, tokens_in_documents)
        """
        if len(inputs.shape) == 2:
            inputs = inputs.unsqueeze(dim=0)
        batch_size = inputs.shape[0]
        tokens_in_document = inputs.shape[1]

        # shape: batch_size, tokens_in_documents, heads_number * key_dim
        keys = torch.matmul(inputs, self.key_weights)
        values = torch.matmul(inputs, self.value_weights)

        # shape: batch_size, tokens_in_documents, heads_number, key_dim
        keys = keys.view(batch_size, -1, self.heads_number, self.key_dim)
        values = values.view(batch_size, -1, self.heads_number, self.key_dim)

        mask = padding_mask.unsqueeze(1)
        if self.is_self_decoder_layer:
            subsequent_mask = torch.ones((tokens_in_document, tokens_in_document), dtype=torch.bool)
            subsequent_mask = torch.triu(subsequent_mask, diagonal=1)
            subsequent_mask = subsequent_mask.unsqueeze(dim=0)
            mask = mask | subsequent_mask

        # shape: batch_size, tokens_in_documents, embedding_dim
        outputs = super().forward(inputs, keys, values, mask)
        return outputs

