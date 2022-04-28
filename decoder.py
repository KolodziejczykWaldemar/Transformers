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
        self.decoder_self_attention = MultiheadedSelfAttention(key_dim=key_dim,
                                                               embedding_dim=embedding_dim,
                                                               heads_number=heads_number,
                                                               is_decoder_layer=True)
        self.layer_norm_0 = LayerNorm(gain=layer_norm_gain)

        self.decoder_encoder_attention = MultiheadedAttention(key_dim=key_dim,
                                                              embedding_dim=embedding_dim,
                                                              heads_number=heads_number)
        self.layer_norm_1 = LayerNorm(gain=layer_norm_gain)

        self.position_wise_dense = PositionWiseDenseNetwork(hidden_dim=hidden_dim,
                                                            embedding_dim=embedding_dim)

        self.layer_norm_2 = LayerNorm(gain=layer_norm_gain)

        self.encoder_keys_weights = torch.rand(size=(self.embedding_dim, self.key_dim * self.heads_number))
        self.encoder_values_weights = torch.rand(size=(self.embedding_dim, self.key_dim * self.heads_number))

    def forward(self, x, encoder_outputs, encoder_padding_mask, decoder_padding_mask):
        self_attention_representations = self.decoder_self_attention(x, decoder_padding_mask)
        x = self.layer_norm_0(x + self_attention_representations)

        # shape: batch_size, encoder_tokens_in_documents, heads_number * key_dim
        encoder_keys = torch.matmul(encoder_outputs, self.encoder_keys_weights)
        encoder_values = torch.matmul(encoder_outputs, self.encoder_values_weights)

        # shape: batch_size, tokens_in_documents, heads_number, key_dim
        encoder_keys = encoder_keys.view(x, -1, self.heads_number, self.key_dim)
        encoder_values = encoder_values.view(x, -1, self.heads_number, self.key_dim)

        encoder_padding_mask = encoder_padding_mask.unsqueeze(1)
        attention_representations = self.decoder_encoder_attention(x, encoder_keys, encoder_values, encoder_padding_mask)

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

    def forward(self, x, encoder_outputs, encoder_padding_mask, decoder_padding_mask):
        for block_id in range(self.blocks_number):
            x = self.decoder_blocks[block_id](x, encoder_outputs, encoder_padding_mask, decoder_padding_mask)

        output_logits = torch.matmul(x, self.output_weights)
        tokens_probs = torch.softmax(output_logits, dim=-1)
        tokens_ids = torch.argmax(tokens_probs, dim=-1)
        return tokens_ids
