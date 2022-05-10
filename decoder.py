import torch
import torch.nn as nn

from attention import MultiheadedAttention
from feed_forward import PositionWiseDenseNetwork, LayerNorm


class DecoderBlock(nn.Module):
    def __init__(self,
                 key_dim: int = 64,
                 embedding_dim: int = 512,
                 heads_number: int = 8,
                 hidden_dim: int = 2048,
                 dropout_prob: float = 0.1) -> None:
        super().__init__()
        self.key_dim = key_dim
        self.heads_number = heads_number
        self.embedding_dim = embedding_dim

        self.decoder_self_attention = MultiheadedAttention(key_dim=key_dim,
                                                           embedding_dim=embedding_dim,
                                                           heads_number=heads_number)
        self.layer_norm_0 = LayerNorm(embedding_dim=embedding_dim)
        self.dropout_0 = nn.Dropout(p=dropout_prob)

        self.decoder_encoder_attention = MultiheadedAttention(key_dim=key_dim,
                                                              embedding_dim=embedding_dim,
                                                              heads_number=heads_number)
        self.layer_norm_1 = LayerNorm(embedding_dim=embedding_dim)
        self.dropout_1 = nn.Dropout(p=dropout_prob)

        self.position_wise_dense = PositionWiseDenseNetwork(hidden_dim=hidden_dim,
                                                            embedding_dim=embedding_dim,
                                                            dropout_prob=dropout_prob)

        self.layer_norm_2 = LayerNorm(embedding_dim=embedding_dim)
        self.dropout_2 = nn.Dropout(p=dropout_prob)

    def forward(self,
                x: torch.Tensor,
                encoder_outputs: torch.Tensor,
                encoder_padding_mask: torch.Tensor,
                decoder_padding_mask: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        tokens_in_document = x.shape[1]

        decoder_mask = decoder_padding_mask.unsqueeze(dim=1).unsqueeze(dim=2)
        subsequent_mask = torch.ones((tokens_in_document, tokens_in_document), dtype=torch.bool)
        subsequent_mask = torch.triu(subsequent_mask, diagonal=1)
        subsequent_mask = subsequent_mask.unsqueeze(dim=0).unsqueeze(dim=1)
        decoder_mask = decoder_mask | subsequent_mask

        self_attention_representations = self.decoder_self_attention(x, x, x, decoder_mask)
        x = self.layer_norm_0(x + self_attention_representations)
        x = self.dropout_0(x)

        encoder_padding_mask = encoder_padding_mask.unsqueeze(dim=1).unsqueeze(dim=2)
        attention_representations = self.decoder_encoder_attention(x, encoder_outputs, encoder_outputs, encoder_padding_mask)

        x = self.layer_norm_1(x + attention_representations)
        x = self.dropout_1(x)

        position_wise_values = self.position_wise_dense(x)
        x = self.layer_norm_2(x + position_wise_values)
        x = self.dropout_2(x)
        return x


class Decoder(nn.Module):
    def __init__(self,
                 vocabulary_size: int,
                 blocks_number: int = 8,
                 key_dim: int = 64,
                 embedding_dim: int = 512,
                 heads_number: int = 8,
                 hidden_dim: int = 2048,
                 dropout_prob: float = 0.1) -> None:
        super().__init__()
        self.blocks_number = blocks_number
        self.decoder_blocks = nn.ModuleList([DecoderBlock(key_dim=key_dim,
                                                          embedding_dim=embedding_dim,
                                                          heads_number=heads_number,
                                                          hidden_dim=hidden_dim,
                                                          dropout_prob=dropout_prob)
                                             for _ in range(self.blocks_number)])
        self.output_weights = nn.Parameter(torch.rand(size=(embedding_dim, vocabulary_size)))
        nn.init.xavier_uniform_(self.output_weights)

    def forward(self,
                x: torch.Tensor,
                encoder_outputs: torch.Tensor,
                decoder_padding_mask: torch.Tensor,
                encoder_padding_mask: torch.Tensor) -> torch.Tensor:
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, encoder_outputs, encoder_padding_mask, decoder_padding_mask)

        output_logits = torch.matmul(x, self.output_weights)
        # we don't apply softmax since loss function does it inplace
        # tokens_probs = torch.softmax(output_logits, dim=-1)
        return output_logits
