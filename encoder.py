import torch.nn as nn

from attention import MultiheadedSelfAttention
from feed_forward import PositionWiseDenseNetwork, LayerNorm


class EncoderBlock(nn.Module):
    def __init__(self,
                 key_dim: int = 64,
                 embedding_dim: int = 512,
                 heads_number: int = 8,
                 hidden_dim: int = 2048,
                 layer_norm_gain: int = 1) -> None:
        super().__init__()
        self.multiheaded_self_attention = MultiheadedSelfAttention(key_dim=key_dim,
                                                                   embedding_dim=embedding_dim,
                                                                   heads_number=heads_number,
                                                                   masked=False)
        self.position_wise_dense = PositionWiseDenseNetwork(hidden_dim=hidden_dim,
                                                            embedding_dim=embedding_dim)

        self.layer_norm = LayerNorm(gain=layer_norm_gain)

    def forward(self, x):
        attention_representations, keys, values = self.multiheaded_self_attention(x)
        x = self.layer_norm(x + attention_representations)
        position_wise_values = self.position_wise_dense(x)
        x = self.layer_norm(x + position_wise_values)
        return x, keys, values


class Encoder(nn.Module):
    def __init__(self,
                 key_dim: int = 64,
                 embedding_dim: int = 512,
                 heads_number: int = 8,
                 hidden_dim: int = 2048,
                 blocks_number: int = 4,
                 layer_norm_gain: int = 1):
        super().__init__()
        self.blocks_number = blocks_number
        self.encoder_blocks = [EncoderBlock(key_dim=key_dim,
                                            embedding_dim=embedding_dim,
                                            heads_number=heads_number,
                                            hidden_dim=hidden_dim,
                                            layer_norm_gain=layer_norm_gain)
                               for _ in range(self.blocks_number)]

    def forward(self, x):
        for block_id in range(self.blocks_number):
            x, keys, values = self.encoder_blocks[block_id](x)
        return x, keys, values