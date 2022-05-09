import torch
import torch.nn as nn

from attention import MultiheadedAttention
from feed_forward import PositionWiseDenseNetwork, LayerNorm


class EncoderBlock(nn.Module):
    def __init__(self,
                 key_dim: int = 64,
                 embedding_dim: int = 512,
                 heads_number: int = 8,
                 hidden_dim: int = 2048,
                 dropout_prob: float = 0.1) -> None:
        super().__init__()
        self.self_attention = MultiheadedAttention(key_dim=key_dim,
                                                   embedding_dim=embedding_dim,
                                                   heads_number=heads_number)
        self.layer_norm_0 = LayerNorm(embedding_dim=embedding_dim)
        self.dropout_0 = nn.Dropout(p=dropout_prob)

        self.position_wise_dense = PositionWiseDenseNetwork(hidden_dim=hidden_dim,
                                                            embedding_dim=embedding_dim,
                                                            dropout_prob=dropout_prob)

        self.layer_norm_1 = LayerNorm(embedding_dim=embedding_dim)
        self.dropout_1 = nn.Dropout(p=dropout_prob)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        encoder_padding_mask = padding_mask.unsqueeze(dim=1).unsqueeze(dim=2)
        attention_representations = self.self_attention(x, x, x, encoder_padding_mask)
        x = self.layer_norm_0(x + attention_representations)
        x = self.dropout_0(x)
        position_wise_values = self.position_wise_dense(x)
        x = self.layer_norm_1(x + position_wise_values)
        x = self.dropout_1(x)
        return x


class Encoder(nn.Module):
    def __init__(self,
                 key_dim: int = 64,
                 embedding_dim: int = 512,
                 heads_number: int = 8,
                 hidden_dim: int = 2048,
                 blocks_number: int = 4,
                 dropout_prob: float = 0.1):
        super().__init__()
        self.blocks_number = blocks_number
        self.encoder_blocks = nn.ModuleList([EncoderBlock(key_dim=key_dim,
                                                          embedding_dim=embedding_dim,
                                                          heads_number=heads_number,
                                                          hidden_dim=hidden_dim,
                                                          dropout_prob=dropout_prob)
                                             for _ in range(self.blocks_number)])

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, padding_mask)
        return x
