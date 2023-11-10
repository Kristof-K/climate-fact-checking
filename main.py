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
    x_num, indices = text_embedding.encode_x(masked_statements)
    y_num, y_num_no_start = text_embedding.encode_y([masked_words[i] for i in indices])

    return x_num, y_num, y_num_no_start


def get_training_data_generator(sentences, batch_size, text_prepro, text_embedding):
    i_sent = 0
    n_sent = len(sentences)
    # get first batch
    new_indices = (i_sent + np.arange(batch_size)) % n_sent
    x_num, y_num, y_num_no_start = get_all_training_data(
        [sentences[i] for i in new_indices], text_prepro, text_embedding
    )
    while True:
        # due to the masked training we get more samples than what we processed
        yield (x_num[:batch_size], y_num[:batch_size]), y_num_no_start[:batch_size]
        x_num_prev = x_num[batch_size:]
        y_num_prev = y_num[batch_size:]
        y_num_no_start_prev = y_num_no_start[batch_size:]

        if x_num_prev.shape[0] < batch_size:
            # process next batch
            i_sent = (i_sent + batch_size) % n_sent
            new_indices = (i_sent + np.arange(batch_size)) % n_sent
            x_num, y_num, y_num_no_start = get_all_training_data(
                [sentences[i] for i in new_indices], text_prepro, text_embedding
            )
            x_num = np.vstack((x_num_prev, x_num))
            y_num = np.vstack((y_num_prev, y_num))
            y_num_no_start = np.vstack((y_num_no_start_prev, y_num_no_start))
        else:
            x_num = x_num_prev
            y_num = y_num_prev
            y_num_no_start = y_num_no_start_prev


if __name__ == '__main__':
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)

    # load and preprocess corpus -------------------------------------------------------------
    corpus = load_corpus(folders=config['preprocessing']['folders'])
    print(f'\nSize corpus: {len(corpus)}')
    text_prepro = TextPreprocessor(config['preprocessing'])
    sentences = text_prepro.extract_sentences(corpus)

    # encode text by word_embedding or character/word one hot encoding -----------------------
    config['encoding']['mask_symbol'] = config['preprocessing']['mask_symbol']
    text_embedding = get_encoder(config['encoding'])
    text_embedding.learn_encoding(sentences)
    vocab_size = text_embedding.get_vocab_size()

    # train or load the model ----------------------------------------------------------------
    model = get_model(text_embedding, vocab_size, config['model_training'])
    if config['model_training']['train']:
        path = create_folder(config['model_training']['model'])     # create folder to store models
        shutil.copy(CONFIG_FILE, path)                              # and copy config file to it

        if config['model_training']['use_generator']:
            # if training data does not fit in the RAM use a generator
            batch_size = config['model_training']['batch_size']
            data_gen = get_training_data_generator(sentences, batch_size, text_prepro, text_embedding)
            model.train_generator(data_gen, steps=len(sentences) // batch_size + 1, path=path)
        else:
            x_num, y_num, y_num_no_start = get_all_training_data(sentences, text_prepro, text_embedding)
            print(f'x-dim: {x_num.shape}\ny-dim: {y_num.shape}')
            model.train(x_num, y_num, y_num_no_start, path=path)

    else:
        model.load_model(path=os.path.join(MODEL_PATH, config['load_model']['folder']),
                         epoch=config['load_model']['epoch'])

    # check performance on some of the training statements -----------------------------------
    pick = random.sample(range(len(sentences)), 10)

    for i in pick:
        masked_statements, masked_words = text_prepro.get_masked_word_tokens([sentences[i]])
        x_num, indices = text_embedding.encode_x(masked_statements)
        masked_words = [masked_words[l] for l in indices]
        if not indices:      # sentences were too long
            continue
        print(f'\n\n#{i}\n{sentences[i]}\n')
        for k in range(len(masked_words)):
            # keep batch dimension by using [k] instead of k
            prob = model.get_token_probability(x_num[[k], :, :], masked_words[k])
            most_likely, prob_ml = model.get_most_likely_words(x_num[[k], :, :], n_beams=3)
            print(f'{masked_words[k]} : {prob:.4f}')
            print('vs. ', end='')
            for l in range(3):
                print(f'{most_likely[l]} : {prob_ml[l]:.4f}, ', end='')
            print()
