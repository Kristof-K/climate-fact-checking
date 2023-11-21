from typing import Iterator, List

import numpy as np
from transformers import AutoTokenizer

from text_encoding.text_encoder import TextEncoder


class MyBertTokenizer(TextEncoder):

    def __init__(self, model_name: str, max_seq_length: int):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_seq_length = min(self.tokenizer.model_max_length, max_seq_length)
        self.max_output = self.tokenizer.vocab_size
        self.mask_token_id = self.tokenizer.mask_token_id
        self.start_token = self.tokenizer.cls_token_id
        self.stop_token = self.tokenizer.sep_token_id
        self.pad_token = self.tokenizer.pad_token_id

    def get_vocab_size(self):
        return self.max_output

    def learn_encoding(self, sentences: Iterator[List[str]]):
        pass    # nothing to do

    def sample_ok(self, sentence: List[str]):
        # BERT breaks unknown tokens down until it knows them, so we just have to check for sequence length
        # BERT uses start and end token
        fine = len(sentence) <= self.max_seq_length - 2
        return fine

    def encode_x(self, samples_x: List[List[str]]):
        # translate token by token, and pad to full length
        ids = [
            [self.start_token] + self.tokenizer.convert_tokens_to_ids(tokens) + [self.stop_token] +
            [self.pad_token] * (self.max_seq_length - 2 - len(tokens))
        for tokens in samples_x]

        return np.array(ids)

    def encode_training_xy(self, samples_x: List[List[str]], samples_y: List[str]):
        x_num = self.encode_x(samples_x)
        y_num = x_num.copy()
        # replace the masked statements with the true labels (undo masking)
        y_num[y_num == self.mask_token_id] = np.array([self.tokenizer.convert_tokens_to_ids(t) for t in samples_y])

        return x_num, y_num

    def encode_one_y(self, y: str):
        return self.tokenizer.convert_tokens_to_ids(y)

    def decode(self, indices: np.array):
        return [self.tokenizer.decode(i) for i in indices]
