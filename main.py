import yaml
import random
import datetime
import os
import shutil   # to copy files
import numpy as np

from utils.load_corpus import load_corpus
from utils.preprocess_corpus import TextPreprocessor
from text_encoding.get_encoder import get_encoder
from models.get_model import get_model

CONFIG_FILE = os.path.join('config', 'run.yaml')
MODEL_PATH = os.path.join('models', 'saved_models')
DATA_PATH = os.path.join('data')


def create_folder(model_name: str):
    ts = datetime.datetime.now()
    ts_str = '{}-{:02d}-{:02d}_{:02d}-{:02d}'.format(
        ts.date().year, ts.date().month, ts.date().day, ts.time().hour, ts.time().minute
    )
    path = os.path.join(MODEL_PATH, model_name + '_' + ts_str)
    os.mkdir(path)
    return path


def get_all_training_data(sentences, text_prepro, text_embedding):
    masked_statements, masked_words = text_prepro.get_masked_word_tokens(sentences)
    x_num = text_embedding.encode_x(masked_statements)
    y_num = text_embedding.encode_y(masked_words)
    return x_num, y_num


def get_training_data_generator(batch_size, text_prepro, text_embedding):
    sent_batch = []
    x_num_prev = None
    y_num_prev = None
    while True:
        sent_generator = text_prepro.get_sent_generator()       # get a new generator

        for sent in sent_generator:
            if not text_embedding.sample_ok(sent):
                continue        # just take the next sentence
            sent_batch.append(sent)

            # process new batch
            if len(sent_batch) >= batch_size:
                x_num, y_num = get_all_training_data(sent_batch, text_prepro, text_embedding)
                if x_num_prev is not None:
                    x_num = np.vstack((x_num_prev, x_num))
                    y_num = np.vstack((y_num_prev, y_num))
                # due to the masked training we get more samples than what we processed
                yield x_num[:batch_size], y_num[:batch_size]
                x_num_prev = x_num[batch_size:]
                y_num_prev = y_num[batch_size:]
                # yield previous processing
                while x_num_prev.shape[0] >= batch_size:
                    yield x_num_prev[:batch_size], y_num_prev[:batch_size]
                    x_num_prev = x_num_prev[batch_size:]
                    y_num_prev = y_num_prev[batch_size:]
                sent_batch = []


if __name__ == '__main__':
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)

    if not os.path.exists(MODEL_PATH):
        os.mkdir(MODEL_PATH)

    # load and preprocess corpus -------------------------------------------------------------
    data_file = os.path.join(DATA_PATH, config['preprocessing']['data_file'])
    config['preprocessing']['data_file'] = data_file
    text_prepro = TextPreprocessor(config['preprocessing'])
    if not os.path.exists(data_file):
        # create assembled sentence file
        corpus_generator = load_corpus(DATA_PATH, folders=config['preprocessing']['folders'])
        text_prepro.save_sentences(corpus_generator)

    # encode text by word_embedding or character/word one hot encoding -----------------------
    config['encoding']['mask_symbol'] = config['preprocessing']['mask_symbol']
    text_embedding = get_encoder(config['encoding'])
    text_embedding.learn_encoding(text_prepro.get_sent_generator())
    num_tokens = text_prepro.get_num_of_tokens()
    print(f'Corpus comprises {num_tokens} climate statements')

    # train or load the model ----------------------------------------------------------------
    model = get_model(text_embedding, config['model_training'])
    if config['model_training']['train']:
        path = create_folder(config['model_training']['model'])     # create folder to store models
        shutil.copy(CONFIG_FILE, path)                              # and copy config file to it

        batch_size = config['model_training']['batch_size']
        data_gen = get_training_data_generator(batch_size, text_prepro, text_embedding)
        model.train(data_gen, steps=num_tokens // batch_size + 1, path=path)

    else:
        model.load_model(path=os.path.join(MODEL_PATH, config['load_model']['folder']),
                         epoch=config['load_model']['epoch'])

    # check performance on some of the training statements -----------------------------------
    pick = [13, 52, 100, 501, 1000]
    sent_gen = text_prepro.get_sent_generator()

    for i, sent in enumerate(sent_gen):
        if i not in pick or not text_embedding.sample_ok(sent):
            continue
        masked_statements, masked_words = text_prepro.get_masked_word_tokens([sent])
        x_num = text_embedding.encode_x(masked_statements)

        print(f'\n\n#{i}\n{" ".join(sent)}\n')
        for k in range(len(masked_words)):
            # keep batch dimension by using [k] instead of k
            prob = model.get_token_probability(x_num[[k], :, :], masked_words[k])
            most_likely, prob_ml = model.get_most_likely_words(x_num[[k], :, :], n_beams=3)
            print(f'{masked_words[k]} : {prob:.4f}')
            print('vs. ', end='')
            for l in range(3):
                print(f'{most_likely[l]} : {prob_ml[l]:.4f}, ', end='')
            print()
