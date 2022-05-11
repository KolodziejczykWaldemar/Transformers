from typing import Tuple, Callable, List

import spacy

from torchtext.legacy.data import Field, BucketIterator
from torchtext.legacy.datasets.translation import Multi30k


class DataLoader:
    def __init__(self,
                 task_extension: Tuple[str, str],
                 init_token: str,
                 eos_token: str,
                 pad_token: str,
                 min_token_freq: int,
                 batch_size: int,
                 device: str):
        self.task_extension = task_extension
        self.init_token = init_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.min_token_freq = min_token_freq
        self.batch_size = batch_size
        self.device = device

        self.spacy_de = spacy.load('de_core_news_sm')
        self.spacy_en = spacy.load('en_core_web_sm')

        self.source = None
        self.target = None

        self.train_data = None
        self.valid_data = None
        self.test_data = None

        self._build_datasets()

    def _build_datasets(self):
        lower = True
        batch_first = True

        if self.task_extension == ('.de', '.en'):
            self.source = Field(tokenize=self._tokenize_de,
                                init_token=self.init_token,
                                eos_token=self.eos_token,
                                pad_token=self.pad_token,
                                lower=lower,
                                batch_first=batch_first)
            self.target = Field(tokenize=self._tokenize_en,
                                init_token=self.init_token,
                                eos_token=self.eos_token,
                                pad_token=self.pad_token,
                                lower=lower,
                                batch_first=batch_first)

        elif self.task_extension == ('.en', '.de'):
            self.source = Field(tokenize=self._tokenize_en,
                                init_token=self.init_token,
                                eos_token=self.eos_token,
                                pad_token=self.pad_token,
                                lower=lower,
                                batch_first=batch_first)
            self.target = Field(tokenize=self._tokenize_de,
                                init_token=self.init_token,
                                eos_token=self.eos_token,
                                pad_token=self.pad_token,
                                lower=lower,
                                batch_first=batch_first)

        else:
            raise Exception(f'Wrong task extension type: {self.task_extension}')

        self.train_data, self.valid_data, self.test_data = Multi30k.splits(exts=self.task_extension,
                                                                           fields=(self.source, self.target))

        self.source.build_vocab(self.train_data, min_freq=self.min_token_freq)
        self.target.build_vocab(self.train_data, min_freq=self.min_token_freq)

    def get_iterators(self):
        train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
            (self.train_data, self.valid_data, self.test_data),
            batch_size=self.batch_size,
            device=self.device
        )
        return train_iterator, valid_iterator, test_iterator

    def _tokenize_de(self, text: str) -> List[str]:
        return [tok.text for tok in self.spacy_de.tokenizer(text)]

    def _tokenize_en(self, text: str) -> List[str]:
        return [tok.text for tok in self.spacy_en.tokenizer(text)]
