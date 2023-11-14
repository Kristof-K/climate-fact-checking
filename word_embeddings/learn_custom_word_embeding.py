import os
import numpy as np
from gensim.models.word2vec import Word2Vec

from utils.load_corpus import load_corpus
from utils.preprocess_corpus import TextPreprocessor

EMBEDDINGS_PATH = 'word_embeddings'


def learn_word_embedding(sentence_tokens: list[list[str]], mask_symbol):
    num_features = 300
    min_word_count = 3
    num_workers = 1
    window_size = 6     # context window
    subsampling = 1e-3  # subsampling for frequent terms

    model = Word2Vec(
        sentence_tokens, workers=num_workers, vector_size=num_features, min_count=min_word_count,
        window=window_size, sample=subsampling
    )

    # Store just the words + their trained embeddings.
    word_vectors = model.wv     # get the keyed vectors
    word_vectors.add_vector(mask_symbol, np.repeat(1.0, num_features))
    word_vectors.save(os.path.join(EMBEDDINGS_PATH, "climate_word2vec.wordvectors"))


if __name__ == '__main__':
    os.chdir("..")     # go one up in root directory, so that data loading works

    mask_symbol = "<mask>"
    folders = ["UnitedNations", "Wikipedia", "NationalGeographic", "TheNewYorkTimes",
               "NationalOceanicAndAtmosphericAdministration"]

    corpus = load_corpus(folders=folders)
    text_prepro = TextPreprocessor({'lower_case': True, 'min_words': 5, 'mask_symbol': mask_symbol})
    sentences = text_prepro.extract_sentences(corpus)

    tokenized_sents = text_prepro.tokenize_sentences(sentences)

    learn_word_embedding(tokenized_sents, mask_symbol=mask_symbol)
