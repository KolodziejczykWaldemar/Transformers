import torch


class TrainingConfig:
    BATCH_SIZE = 128
    TOTAL_EPOCHS = 1000
    WARMUP_EPOCHS = 100
    INIT_LR = 1e-5
    ADAM_WEIGHT_DECAY = 5e-4
    ADAM_EPSILON = 5e-9
    REDUCE_LR_PATIENCE = 10
    REDUCE_LR_FACTOR = 0.9
    GRADIENT_CLIP = 1.0


class TransformerConfig:
    BATCH_SIZE = 128
    MAX_SOURCE_LEN = 256
    EMBEDDING_DIM = 512
    LATENT_KEY_DIM = 64
    BLOCKS_NUM = 6
    HEADS_NUM = 8
    HIDDEN_FC_DIM = 2048
    DROPOUT_PROB = 0.1
    MIN_TOKEN_FREQ = 2
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SpecialToken:
    PAD_WORD = '<BLK>'
    UNK_WORD = '<UNK>'
    SOS_WORD = '<SOS>'
    EOS_WORD = '<EOS>'
