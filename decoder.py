import torch
import torch.nn as nn

from attention import MultiheadedSelfAttention, MultiheadedAttention
from feed_forward import PositionWiseDenseNetwork, LayerNorm


class DecoderBlock(nn.Module):
    def __init__(self,
                 key_dim: int = 64,
                 embedding_dim: int = 512,
                 heads_number: int = 8,
                 hidden_dim: int = 2048,
                 layer_norm_gain: int = 1) -> None:
        self.multiheaded_masked_self_attention = MultiheadedSelfAttention(key_dim=key_dim,
                                                                          embedding_dim=embedding_dim,
                                                                          heads_number=heads_number,
                                                                          masked=True)
        self.layer_norm_0 = LayerNorm(gain=layer_norm_gain)

        self.multiheaded_attention = MultiheadedAttention(key_dim=key_dim,
                                                          embedding_dim=embedding_dim,
                                                          heads_number=heads_number,
                                                          masked=False)
        self.layer_norm_1 = LayerNorm(gain=layer_norm_gain)

        self.position_wise_dense = PositionWiseDenseNetwork(hidden_dim=hidden_dim,
                                                            embedding_dim=embedding_dim)

        self.layer_norm_2 = LayerNorm(gain=layer_norm_gain)

    def forward(self, x, encoder_keys, encoder_values):
        self_attention_representations, _, __ = self.multiheaded_masked_self_attention(x)
        x = self.layer_norm_0(x + self_attention_representations)
        attention_representations, _, __ = self.multiheaded_attention(x, encoder_keys, encoder_values)
        x = self.layer_norm(x + attention_representations)
        position_wise_values = self.position_wise_dense(x)
        x = self.layer_norm(x + position_wise_values)
        return x


class Decoder(nn.Module):
    def __init__(self,
                 vocabulary_size: int,
                 blocks_number: int = 8,
                 key_dim: int = 64,
                 embedding_dim: int = 512,
                 heads_number: int = 8,
                 hidden_dim: int = 2048,
                 layer_norm_gain: int = 1) -> None:
        self.blocks_number = blocks_number
        self.decoder_blocks = [DecoderBlock(key_dim=key_dim,
                                            embedding_dim =embedding_dim,
                                            heads_number=heads_number,
                                            hidden_dim=hidden_dim,
                                            layer_norm_gain=layer_norm_gain)
                               for _ in range(self.blocks_number)]
        self.output_weights = torch.rand(size=(embedding_dim, vocabulary_size))

    def forward(self, x, encoder_keys, encoder_values):
        for block_id in range(self.blocks_number):
            x = self.decoder_blocks[block_id](x, encoder_keys, encoder_values)

        output_logits = torch.matmul(x, self.output_weights)
        tokens_probs = torch.softmax(output_logits, dim=-1)
        tokens_ids = torch.argmax(tokens_probs, dim=-1)
        return tokens_ids
