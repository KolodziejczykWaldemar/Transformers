import math
import os
from time import sleep

import torch
from torch import nn, optim
from tqdm import tqdm

from config import TrainingConfig, SpecialToken
from dataset import DataLoader


class Trainer:
    def __init__(self, loader: DataLoader) -> None:
        self.loader = loader

        self.train_data_iter, self.valid_data_iter, self.test_data_iter = self.loader.get_iterators()

        self.criterion = None
        self.optimizer = None
        self.scheduler = None

    def train(self, model: nn.Module, total_epochs: int) -> None:
        best_loss = float('inf')

        for current_epoch in range(total_epochs):
            train_loss = self._train_step(model, current_epoch)
            valid_loss = self._eval_step(model)

            if current_epoch > TrainingConfig.WARMUP_EPOCHS:
                self.scheduler.step(valid_loss)

            if valid_loss < best_loss:
                os.makedirs('saved', exist_ok=True)
                torch.save(model.state_dict(), f'saved/model-{valid_loss}.pt')

                best_loss = valid_loss

            print(f'\n\tTrain loss: {train_loss:.3f}')
            print(f'\tValid loss: {valid_loss:.3f}')

    def test(self, model: nn.Module) -> None:
        model.eval()

        with torch.no_grad():
            for batch in self.test_data_iter:
                src = batch.src
                trg = batch.trg
                output = model(src, trg[:, :-1])

                for i in range(src.shape[0]):
                    src_words = " ".join([self.loader.source.vocab.itos[token_id] for token_id in src[i]])
                    trg_words = " ".join([self.loader.target.vocab.itos[token_id] for token_id in trg[i]])

                    output_words = output[i].max(dim=1)[1]
                    output_words = " ".join([self.loader.target.vocab.itos[token_id] for token_id in output_words])

                    print('source: \t', src_words)
                    print('target: \t', trg_words)
                    print('predicted: \t', output_words)
                    print()

    def _setup_objective(self, model: nn.Module) -> None:
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.loader.target.vocab.stoi[SpecialToken.PAD_WORD])
        self.optimizer = optim.Adam(params=model.parameters(),
                                    lr=TrainingConfig.INIT_LR,
                                    weight_decay=TrainingConfig.ADAM_WEIGHT_DECAY,
                                    eps=TrainingConfig.ADAM_EPSILON)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer,
                                                              verbose=True,
                                                              factor=TrainingConfig.REDUCE_LR_FACTOR,
                                                              patience=TrainingConfig.REDUCE_LR_PATIENCE)

    def _train_step(self, model: nn.Module, epoch_number: int) -> float:
        model.train()
        self._setup_objective(model)

        epoch_loss = 0
        with tqdm(self.train_data_iter, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch_number}")
                src = batch.src
                trg = batch.trg

                self.optimizer.zero_grad()
                output = model(src, trg[:, :-1])
                output_reshape = output.contiguous().view(-1, output.shape[-1])
                trg = trg[:, 1:].contiguous().view(-1)

                loss = self.criterion(output_reshape, trg)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), TrainingConfig.GRADIENT_CLIP)
                self.optimizer.step()

                epoch_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
                sleep(0.1)

        return epoch_loss / len(self.train_data_iter)

    def _eval_step(self, model: nn.Module) -> float:
        model.eval()

        epoch_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(self.valid_data_iter):
                src = batch.src
                trg = batch.trg
                output = model(src, trg[:, :-1])
                output_reshape = output.contiguous().view(-1, output.shape[-1])
                trg = trg[:, 1:].contiguous().view(-1)

                loss = self.criterion(output_reshape, trg)
                epoch_loss += loss.item()

        return epoch_loss / len(self.valid_data_iter)
