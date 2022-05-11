import torch

from config import TransformerConfig, SpecialToken, TrainingConfig
from dataset import DataLoader
from trainer import Trainer
from transformer import Transformer

if __name__ == '__main__':
    loader = DataLoader(task_extension=('.en', '.de'),
                        init_token=SpecialToken.SOS_WORD,
                        eos_token=SpecialToken.EOS_WORD,
                        pad_token=SpecialToken.PAD_WORD,
                        min_token_freq=TransformerConfig.MIN_TOKEN_FREQ,
                        batch_size=TransformerConfig.BATCH_SIZE,
                        device=TransformerConfig.DEVICE
                        )

    model = Transformer(enc_pad_token_id=loader.source.vocab.stoi[SpecialToken.PAD_WORD],
                        dec_sos_token_id=loader.target.vocab.stoi[SpecialToken.SOS_WORD],
                        dec_pad_token_id=loader.target.vocab.stoi[SpecialToken.PAD_WORD],
                        dec_eos_token_id=loader.target.vocab.stoi[SpecialToken.EOS_WORD],
                        enc_vocab_size=len(loader.source.vocab),
                        dec_vocab_size=len(loader.target.vocab),
                        embedding_dim=TransformerConfig.EMBEDDING_DIM,
                        blocks_number=TransformerConfig.BLOCKS_NUM,
                        key_dim=TransformerConfig.LATENT_KEY_DIM,
                        heads_number=TransformerConfig.HEADS_NUM,
                        hidden_dim=TransformerConfig.HIDDEN_FC_DIM,
                        dropout_prob=TransformerConfig.DROPOUT_PROB,
                        max_doc_len=TransformerConfig.MAX_SOURCE_LEN)
    model.initialize_weights()

    trainer = Trainer(loader=loader)
    trainer.train(model=model, total_epochs=TrainingConfig.TOTAL_EPOCHS)

    # For testing, uncomment the following line and provide proper path to the saved model
    # model.load_state_dict(torch.load("./saved/model-4.733623266220093.pt"))
    # trainer.test(model=model)