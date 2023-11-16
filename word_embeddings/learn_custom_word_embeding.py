import os

import numpy as np
from gensim.models.word2vec import Word2Vec

from utils.load_corpus import load_corpus
from utils.preprocess_corpus import TextPreprocessor

EMBEDDINGS_PATH = 'word_embeddings'
DATA_PATH = os.path.join('data')


class MyCorpus:
    """An iterator that yields sentences (lists of str)."""
    # just do it as on: https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html

    def __init__(self, text_prepro):
        self.text_prepro = text_prepro

    def __iter__(self):
        tokenized_sentences = self.text_prepro.get_sent_generator()
        for line in tokenized_sentences:
            yield line


def learn_word_embedding(sentence_tokens, mask_symbol):
    num_features = 300
    min_word_count = 3
    num_workers = 1
    window_size = 6     # context window
    subsampling = 1e-3  # subsampling for frequent terms

    model = Word2Vec(
        sentences=sentence_tokens, workers=num_workers, vector_size=num_features, min_count=min_word_count,
        window=window_size, sample=subsampling
    )

    # Store just the words + their trained embeddings.
    word_vectors = model.wv     # get the keyed vectors
    word_vectors.add_vector(mask_symbol, np.repeat(1.0, num_features))
    word_vectors.save(os.path.join(EMBEDDINGS_PATH, "climate_word2vec.wordvectors"))
    print(f'Found {len(word_vectors.vectors)} word vectors')


def analyze_sentences(sentences):
    # analyze sentence characteristics: we have to use statistics that can be calculated on a stream of data as we
    # cannot read in all the data at once
    sent_lengths = np.array([25, 50, 75, 100, 150, 200, 300, 400, 500])
    sent_counts = np.zeros(sent_lengths.size + 1)
    sent_length_mean = 0.0
    sent_count = 0
    word_lengths = np.array([2, 4, 6, 8, 10, 15, 20, 25])
    word_counts = np.zeros(word_lengths.size + 1)
    word_length_mean = 0.0
    word_count = 0

    for sent in sentences:
        # update sentence summary statistics
        sent_count += 1
        new_length = len(' '.join(sent))
        sent_length_mean = sent_length_mean * (sent_count - 1) / sent_count + new_length / sent_count
        sent_counts[np.sum(new_length >= sent_lengths)] += 1

        for w in sent:
            word_count += 1
            word_length_mean = word_length_mean * (word_count - 1) / word_count + len(w) / word_count
            word_counts[np.sum(len(w) >= word_lengths)] += 1

    print('Sentence statistics')
    print(f'Mean: {sent_length_mean}')
    print("Counts:", sent_lengths)
    print(sent_counts)
    print('\nWord statistics')
    print(f'Mean: {word_length_mean}')
    print("Counts:", word_lengths)
    print(word_counts)


if __name__ == '__main__':
    os.chdir("..")     # go one up in root directory, so that data loading works

    mask_symbol = "<mask>"
    folders = ["UnitedNations", "Wikipedia", "NationalGeographic", "TheNewYorkTimes",
               "NationalOceanicAndAtmosphericAdministration"]

    corpus_generator = load_corpus(DATA_PATH, folders=folders)
    text_prepro = TextPreprocessor({'lower_case': True, 'min_words': 5, 'mask_symbol': mask_symbol,
                                    'data_file': os.path.join(DATA_PATH, 'assembled_statements.txt')})
    text_prepro.save_sentences(corpus_generator)

    analyze_sentences(text_prepro.get_sent_generator())

    learn_word_embedding(MyCorpus(text_prepro), mask_symbol=mask_symbol)
