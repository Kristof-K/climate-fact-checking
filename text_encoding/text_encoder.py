from abc import ABC, abstractmethod
from typing import Iterator, List

import numpy as np


class TextEncoder(ABC):

    @abstractmethod
    def get_vocab_size(self):
        pass

    @abstractmethod
    def learn_encoding(self, sentences: Iterator[List[str]]):
        pass

    @abstractmethod
    def sample_ok(self, sentences: List[str]):
        pass

    @abstractmethod
    def encode_x(self, samples_x: List[List[str]]):
        pass

    @abstractmethod
    def encode_training_xy(self, samples_x: List[List[str]], samples_y: List[str]):
        pass

    @abstractmethod
    def encode_one_y(self, sample_y: str):
        pass

    @abstractmethod
    def decode(self, indices: np.array):
        pass
