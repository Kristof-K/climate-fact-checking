import numpy as np
import sys

from text_encoding.text_encoder import TextEncoder


class CharOneHotEncoder(TextEncoder):

    def __init__(self, max_seq_length: int, start_token: str, stop_token: str):
        self.max_seq_length = max_seq_length
        self.start_token = start_token
        self.stop_token = stop_token
        self.char_to_index = {}
        self.index_to_char = {}
        self.vocab_size = 0
        self.max_output = 0

    def get_vocab_size(self):
        return self.vocab_size

    def learn_encoding(self, samples_x: list[list[str]], samples_y: list[str]):
        # determine list of all used characters
        # due to masked training samples_x and samples_y contain the same characters
        # except for the masking symbol
        chars = list(set(' '.join(samples_y)))
        for char in ' '.join(samples_x[0]):
            if char not in chars:
                chars.append(char)
        chars.append(self.start_token)
        chars.append(self.stop_token)
        chars = sorted(chars)

        self.char_to_index = dict((c, i) for i, c in enumerate(chars))
        self.index_to_char = dict((i, c) for i, c in enumerate(chars))
        self.vocab_size = len(chars)
        # we are adding start and stop token to the output string
        self.max_output = max([len(word) for word in samples_y]) + 2

    def encode_x(self, samples_x: list[list[str]]):
        if self.char_to_index == {}:
            print('learn_and_encode() has to be invoked first', file=sys.stderr)

        # throw out too long samples
        indices = [i for i in range(len(samples_x)) if len(' '.join(samples_x[i])) <= self.max_seq_length]
        x_num = np.zeros((len(indices), self.max_seq_length, self.vocab_size), dtype='float32')

        for i, index in enumerate(indices):
            for t, char in enumerate(' '.join(samples_x[index])):
                x_num[i, t, self.char_to_index[char]] = 1.0
        return x_num, indices

    def encode_y(self, samples_y: list[str]):
        if self.char_to_index == {}:
            print('learn_and_encode() has to be invoked first', file=sys.stderr)

        y_num = np.zeros((len(samples_y), self.max_output, self.vocab_size), dtype='float32')
        y_num_no_start = np.zeros((len(samples_y), self.max_output, self.vocab_size), dtype='float32')

        for i, word_token in enumerate(samples_y):
            for t, char in enumerate((self.start_token + word_token + self.stop_token)):
                y_num[i, t, self.char_to_index[char]] = 1.0
        y_num_no_start[:, :-1, :] = y_num[:, 1:, :]        # just forget the start token

        return y_num, y_num_no_start

    def decode(self, model_output: np.array):
        if self.index_to_char == {}:
            print('learn_and_encode() has to be invoked first', file=sys.stderr)

        return self.index_to_char[np.argmax(model_output, axis=0)]
