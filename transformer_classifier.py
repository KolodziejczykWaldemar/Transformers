import torch
import torch.nn as nn

from embedding import TransformerEmbedding
from encoder import Encoder


class TransformerClassifier(nn.Module):
    def __init__(self,
                 classes_number: int,
                 enc_sos_token_id: int,
                 enc_pad_token_id: int,
                 enc_vocab_size: int,
                 embedding_dim: int = 512,
                 blocks_number: int = 8,
                 key_dim: int = 64,
                 heads_number: int = 8,
                 hidden_dim: int = 2048,
                 layer_norm_gain: int = 1):
        super().__init__()
        self.enc_sos_token_id = enc_sos_token_id
        self.enc_pad_token_id = enc_pad_token_id

        self.enc_embeddings = TransformerEmbedding(enc_vocab_size, embedding_dim)
        self.encoder = Encoder(key_dim=key_dim,
                               embedding_dim=embedding_dim,
                               heads_number=heads_number,
                               hidden_dim=hidden_dim,
                               blocks_number=blocks_number,
                               layer_norm_gain=layer_norm_gain)
        self.final_linear = nn.Linear(embedding_dim, classes_number)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        padding_mask = self._get_padding_mask(inputs, self.enc_pad_token_id)

        inputs_emb = self.enc_embeddings(inputs)

        outputs = self.encoder(inputs_emb,
                               padding_mask)
        summed_outputs = torch.sum(outputs, dim=1, keepdim=False)
        return self.final_linear(summed_outputs)

    def _get_padding_mask(self, documents: torch.Tensor, pad_token_id: int) -> torch.Tensor:
        """
        documents.shape = (batch_size, tokens_in_documents)
        """
        mask = documents == pad_token_id
        if len(mask.shape) == 1:
            mask = mask.unsqueeze(0)
        return mask
