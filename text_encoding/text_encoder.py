from abc import ABC, abstractmethod
from typing import Iterator

import numpy as np


class TextEncoder(ABC):

    @abstractmethod
    def get_vocab_size(self):
        pass

    @abstractmethod
    def learn_encoding(self, sentences: Iterator[list[str]]):
        pass

    @abstractmethod
    def sample_ok(self, sentences: list[str]):
        pass

    @abstractmethod
    def encode_x(self, samples_x: list[list[str]]):
        pass

    @abstractmethod
    def encode_y(self, samples_y: list[str]):
        pass

    @abstractmethod
    def encode_one_y(self, sample_y: str):
        pass

    @abstractmethod
    def decode(self, model_output: np.array):
        pass
