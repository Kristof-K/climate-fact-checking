import numpy as np
import sys

from text_encoding.text_encoder import TextEncoder


class CharOneHotEncoder(TextEncoder):

    def __init__(self, max_seq_length: int, max_output: int, start_token: str, stop_token: str,
                 mask_symbol: str):
        self.max_seq_length = max_seq_length
        self.max_output = max_output
        self.start_token = start_token
        self.stop_token = stop_token
        self.mask_symbol = mask_symbol
        self.char_to_index = {}
        self.index_to_char = {}
        self.vocab_size = 0

    def get_vocab_size(self):
        return self.vocab_size

    def learn_encoding(self, sentences: list[str]):
        # determine list of all used characters
        chars = list(set(' '.join(sentences)))
        chars.append(self.mask_symbol)
        chars.append(self.start_token)
        chars.append(self.stop_token)
        if '"' in chars and '`' not in chars:       # nltk.word_tokenize changes double quotes
            chars.append('`')
        if '"' in chars and '\'' not in chars:      # into backward and forward quotes
            chars.append('\'')
        chars = sorted(chars)

        self.char_to_index = dict((c, i) for i, c in enumerate(chars))
        self.index_to_char = dict((i, c) for i, c in enumerate(chars))
        self.vocab_size = len(chars)

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

    def encode_one_y(self, y: str):
        if self.char_to_index == {}:
            print('learn_and_encode() has to be invoked first', file=sys.stderr)

        y_num = np.zeros((1, 1, self.vocab_size), dtype='float32')
        y_num[0, 0, self.char_to_index[y]] = 1.0
        return y_num

    def decode(self, model_output: np.array):
        if self.index_to_char == {}:
            print('learn_and_encode() has to be invoked first', file=sys.stderr)

        return self.index_to_char[np.argmax(model_output, axis=0)]
