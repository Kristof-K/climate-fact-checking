import re   # regular expressions
from typing import Iterator

from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np

# nltk.download('punkt')

NEW_LINE = '\n'


class TextPreprocessor:

    def __init__(self, config: dict):
        self.mask_symbol = config['mask_symbol']
        self.lower_case = config['lower_case']
        self.min_words = config['min_words']
        self.data_file = config['data_file']

    def preprocess_corpus(self, corpus: str):
        # remove artifacts: citation numbers
        citation = r'\[\d+\]'
        corpus_modified = re.sub(citation, '', corpus)
        # remove artifacts: html paragraphs
        corpus_modified = re.sub(r'<p>', '\n', corpus_modified)
        corpus_modified = re.sub(r'</p>', '\n', corpus_modified)
        corpus_modified = re.sub(r'<br>', '', corpus_modified)

        if self.lower_case:
            corpus_modified = corpus_modified.lower()
        return corpus_modified

    def extract_sentences(self, corpus: str):
        corpus_modified = self.preprocess_corpus(corpus)
        # replace newlines and multiple spaces by one space
        corpus_modified = re.sub(r'\n{2,}', '. ', corpus_modified)
        corpus_modified = re.sub(r'\n', ' ', corpus_modified)
        corpus_modified = re.sub(r'\s{2,}', ' ', corpus_modified)
        corpus_modified = re.sub(r'\.{2,}', '.', corpus_modified)
        # sent tokenize does not split sentences if they end with a year number
        # --> insert space to accommodate for that
        corpus_modified = re.sub(r'(\d{4})\. ', r'\1 . ', corpus_modified)
        # break text in statements / sentences
        sentences = sent_tokenize(corpus_modified)
        # filter for non-empty sentences, and remove leading and trailing spaces and point
        sentences = [sentence.strip(' ').rstrip('.') for sentence in sentences if sentence != '' and sentence != '.']

        return sentences

    def save_sentences(self, corpus_generator: Iterator[str]):
        # preprocess sentences of corpus and write preprocessed sentences in one big text file
        with open(self.data_file, 'w', encoding='utf-8') as f:
            for text in corpus_generator:
                sentences = self.extract_sentences(text)
                for sent in sentences:
                    f.write(sent)
                    f.write(NEW_LINE)

    def get_num_of_sentences(self):
        lines = 0
        with open(self.data_file, 'r', encoding='utf-8') as f:
            for line in f:
                lines = lines + 1
        return lines

    def get_sent_generator(self):
        # generator for the preprocessed sentences: yield sentence after sentence and work_tokenize it
        with open(self.data_file, 'r', encoding='utf-8') as f:
            for line in f:      # read file line by line
                word_tokens = word_tokenize(line.rstrip(NEW_LINE))
                if len(word_tokens) >= self.min_words:
                    yield word_tokens

    def get_masked_word_tokens(self, sentences_tokenized: list[list[str]]):
        masked_sentences = []
        masked_word = []

        for word_tokens in sentences_tokenized:
            if len(word_tokens) < self.min_words:
                continue
            # iterate through word tokens and mask each one out
            for i in range(len(word_tokens)):
                masked_sentences.append(word_tokens[0:i] + [self.mask_symbol] + word_tokens[(i+1):len(word_tokens)])
            masked_word += word_tokens

        return masked_sentences, masked_word
