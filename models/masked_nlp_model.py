from abc import ABC, abstractmethod
from typing import Iterator
import numpy as np
import matplotlib.pyplot as plt
import os


class MaskedNLPModel(ABC):
    # abstract class all models should inherit

    @abstractmethod
    def train(self, generator: Iterator, steps: int, path: str):
        pass

    @abstractmethod
    def get_token_probability(self, x_num: np.array, masked_word: str):
        pass

    @abstractmethod
    def get_most_likely_words(self, x_num: np.array, n_beams: int):
        pass

    @abstractmethod
    def load_model(self, folder_name: str, epoch: int):
        pass

    @staticmethod
    def print_loss(loss_vals: dict, path: str):
        plt.plot(loss_vals['loss'])
        plt.title('Loss Values')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.tight_layout()
        plt.savefig(os.path.join(path, 'loss.png'))
