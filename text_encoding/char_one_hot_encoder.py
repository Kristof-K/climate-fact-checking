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

    def learn_encoding(self, sentences: list[list[str]]):
        # determine list of all used characters
        chars = list(set(''.join([''.join(sent) for sent in sentences])))
        chars.append(' ')
        chars.append(self.mask_symbol)
        chars.append(self.start_token)
        chars.append(self.stop_token)
        chars = sorted(chars)

        self.char_to_index = dict((c, i) for i, c in enumerate(chars))
        self.index_to_char = dict((i, c) for i, c in enumerate(chars))
        self.vocab_size = len(chars)

    def _check_word(self, word: str):
        # each character of the word is contained in our vocabulary
        return np.all(np.array([char in self.char_to_index.keys() for char in word]))

    def filter_samples(self, sentences: list[list[str]]):
        if self.char_to_index == {}:
            print('learn_encoding() has to be invoked first', file=sys.stderr)

        # check whether sentence is not too long and all words are contained in our vocabulary
        all_fine = np.array([len(' '.join(sent)) <= self.max_seq_length and
                             np.all([self._check_word(w) for w in sent])
                             for sent in sentences])
        print(f'{all_fine.mean() * 100:.2f}% are suitable for training')
        return [sentences[i] for i in np.arange(len(sentences))[all_fine]]

    def encode_x(self, samples_x: list[list[str]]):
        if self.char_to_index == {}:
            print('learn_encoding() has to be invoked first', file=sys.stderr)

        x_num = np.zeros((len(samples_x), self.max_seq_length, self.vocab_size), dtype='float32')

        for i, sample in enumerate(samples_x):
            for t, char in enumerate(' '.join(sample)):
                x_num[i, t, self.char_to_index[char]] = 1.0
        return x_num

    def encode_y(self, samples_y: list[str]):
        if self.char_to_index == {}:
            print('learn_encoding() has to be invoked first', file=sys.stderr)

        y_num = np.zeros((len(samples_y), self.max_output, self.vocab_size), dtype='float32')

        for i, word_token in enumerate(samples_y):
            for t, char in enumerate((self.start_token + word_token + self.stop_token)):
                y_num[i, t, self.char_to_index[char]] = 1.0

        return y_num

    def encode_one_y(self, y: str):
        if self.char_to_index == {}:
            print('learn_encoding() has to be invoked first', file=sys.stderr)

        y_num = np.zeros((1, 1, self.vocab_size), dtype='float32')
        y_num[0, 0, self.char_to_index[y]] = 1.0
        return y_num

    def decode(self, model_output: np.array):
        if self.index_to_char == {}:
            print('learn_encoding() has to be invoked first', file=sys.stderr)

        return self.index_to_char[np.argmax(model_output, axis=0)]
