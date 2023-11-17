from typing import Iterator, List

import numpy as np
import os
import sys
from gensim.models.keyedvectors import KeyedVectors

from text_encoding.text_encoder import TextEncoder

WORD_VECS_DIR = os.path.join('word_embeddings')
WORD_VECS = 'climate_word2vec.wordvectors'


class MyWord2Vec(TextEncoder):

    def __init__(self, max_seq_length: int, max_output: int, mask_symbol: str):
        self.path_to_word_vecs = os.path.join(WORD_VECS_DIR, WORD_VECS)
        self.max_seq_length = max_seq_length
        self.max_output = max_output
        self.mask_symbol = mask_symbol
        self.word_to_index = {}
        self.index_to_word = {}
        self.word_vectors = None
        self.vec_dim = 0

    def get_vocab_size(self):
        return self.max_output

    def learn_encoding(self, sentences: Iterator[List[str]]):
        if not os.path.exists(self.path_to_word_vecs):
            print(f'The word embedding {self.path_to_word_vecs} does not exist', file=sys.stderr)

        # load most frequent words
        self.word_vectors = KeyedVectors.load(self.path_to_word_vecs, mmap='r')
        # somehow adding does not work
        # self.word_vectors.add_vector(self.mask_symbol, np.repeat(1.0, self.vec_dim))
        self.vec_dim = self.word_vectors.__dict__['vector_size']
        words = self.word_vectors.__dict__['key_to_index'].keys()
        self.max_output = len(words)

        self.word_to_index = dict((c, i) for i, c in enumerate(words))
        self.index_to_word = dict((i, c) for i, c in enumerate(words))

    def sample_ok(self, sentence: List[str]):
        if self.word_vectors is None:
            print('learn_encoding() has to be invoked first', file=sys.stderr)

        # check whether sentence is short enough and all words are contained in our vocabulary
        fine = (len(sentence) <= self.max_seq_length and
                np.all(np.array([token in self.word_to_index.keys() for token in sentence])))
        return fine

    def encode_x(self, samples_x: List[List[str]]):
        if self.word_vectors is None:
            print('learn_encoding() has to be invoked first', file=sys.stderr)

        x_num = np.zeros((len(samples_x), self.max_seq_length, self.vec_dim), dtype='float32')

        for i, sample in enumerate(samples_x):
            for t, word in enumerate(sample):
                x_num[i, t, :] = self.word_vectors[word]
        return x_num

    def encode_y(self, samples_y: List[str]):
        if self.word_vectors is None:
            print('learn_encoding() has to be invoked first', file=sys.stderr)

        y_num = np.zeros((len(samples_y), self.max_output), dtype='float32')

        for i, word_token in enumerate(samples_y):
            y_num[i, self.word_to_index[word_token]] = 1.0

        return y_num

    def encode_one_y(self, y: str):
        if self.word_vectors is None:
            print('learn_encoding() has to be invoked first', file=sys.stderr)

        y_num = np.zeros((1, 1, self.max_output), dtype='float32')
        y_num[0, 0, self.word_to_index[y]] = 1.0
        return y_num

    def decode(self, model_output: np.array):
        if self.word_vectors is None:
            print('learn_encoding() has to be invoked first', file=sys.stderr)

        return self.index_to_word[np.argmax(model_output, axis=0)]
