from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import os


class MaskedNLPModel(ABC):

    @abstractmethod
    def train(self, samples_x: np.array, samples_y: np.array, samples_y_no_start: np.array,
              path: str):
        pass

    @abstractmethod
    def get_token_probability(self, x_num: np.array, masked_word: str):
        pass

    @abstractmethod
    def get_most_likely_word(self, x_num: np.array):
        pass

    @abstractmethod
    def load_model(self, folder_name: str, epoch: int):
        pass

    @staticmethod
    def print_loss(loss_vals: dict, path: str):
        plt.plot(loss_vals['loss'])
        plt.plot(loss_vals['val_loss'])
        plt.title('Loss Values')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(path, 'loss.png'))
