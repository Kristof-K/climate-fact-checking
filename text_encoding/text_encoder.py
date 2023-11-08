from abc import ABC, abstractmethod


class TextEncoder(ABC):

    @abstractmethod
    def get_vocab_size(self):
        pass

    @abstractmethod
    def learn_encoding(self, samples_x, samples_y):
        pass

    @abstractmethod
    def encode_x(self, samples_x):
        pass

    def encode_y(self, samples_y):
        pass

    @abstractmethod
    def decode(self, model_output):
        pass
