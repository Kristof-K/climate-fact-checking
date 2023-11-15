import re   # regular expressions
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np

# nltk.download('punkt')


class TextPreprocessor:

    def __init__(self, config: dict):
        self.mask_symbol = config["mask_symbol"]
        self.lower_case = config["lower_case"]
        self.min_words = config["min_words"]

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

        TextPreprocessor.analyze_sentence_lengths(sentences)

        return sentences

    @staticmethod
    def _analyze_lengths(lengths: np.array, desc: str, q_levels: np.array = np.array([0.1, 0.25, 0.5, 0.75, 0.9])):
        print(f"\n{desc} length distribution:")
        print(f"Min:  {lengths.min()}\nMean: {lengths.mean()}\nMax:  {lengths.max()}")
        print("Quantiles:", q_levels)
        print(np.quantile(lengths, q=q_levels))

    @staticmethod
    def analyze_sentence_lengths(sentences: list[str]):
        lengths = np.array([len(s) for s in sentences])
        TextPreprocessor._analyze_lengths(lengths, "Sentence")

    def tokenize_sentences(self, sentences: list[str]):
        tokenized_sentences = []

        for sent in sentences:
            word_tokens = word_tokenize(sent)

            if len(word_tokens) < self.min_words:
                continue
            tokenized_sentences.append(word_tokens)

        TextPreprocessor.analyze_word_lengths(tokenized_sentences)
        return tokenized_sentences

    @staticmethod
    def analyze_word_lengths(sentences: list[list[str]]):
        # sum appends all lists contained in sentences
        lengths = np.array([len(w) for w in sum(sentences, [])])
        TextPreprocessor._analyze_lengths(lengths, "Word")

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
