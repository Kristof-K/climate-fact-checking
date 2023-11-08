import yaml
import random
import datetime
import os
import shutil   # to copy files

from utils.load_corpus import load_corpus
from utils.preprocess_corpus import TextPreprocessor
from text_encoding.get_encoder import get_encoder
from models.get_model import get_model

CONFIG_FILE = os.path.join('config', 'run.yaml')
MODEL_PATH = os.path.join('models', 'saved_models')


def create_folder(model_name):
    ts = datetime.datetime.now()
    ts_str = '{}-{:02d}-{:02d}_{:02d}-{:02d}'.format(
        ts.date().year, ts.date().month, ts.date().day, ts.time().hour, ts.time().minute
    )
    path = os.path.join(MODEL_PATH, model_name + '_' + ts_str)
    os.mkdir(path)
    return path


if __name__ == '__main__':
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)

    # load and preprocess corpus -------------------------------------------------------------
    corpus = load_corpus(folders=config['preprocessing']['folders'])
    print(f'\nSize corpus: {len(corpus)}')
    text_prepro = TextPreprocessor(config['preprocessing'])
    sentences = text_prepro.extract_sentences(corpus)
    masked_statements, masked_words = text_prepro.get_masked_word_tokens(sentences)

    # encode text by word_embedding or character/word one hot encoding -----------------------
    text_embedding = get_encoder(config['encoding'])
    text_embedding.learn_encoding(masked_statements, masked_words)
    x_num, indices = text_embedding.encode_x(masked_statements)
    y_num, y_num_no_start = text_embedding.encode_y([masked_words[i] for i in indices])
    vocab_size = text_embedding.get_vocab_size()

    # train or load the model ----------------------------------------------------------------
    model = get_model(text_embedding, vocab_size, config['model_training'])
    if config['model_training']['train']:
        path = create_folder(config['model_training']['model'])     # create folder to store models
        shutil.copy(CONFIG_FILE, path)                              # and copy config file to it

        print(f'x-dim: {x_num.shape}\ny-dim: {y_num.shape}')
        model.train(x_num, y_num, y_num_no_start, path)
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
            most_likely, prob_ml = model.get_most_likely_word(x_num[[k], :, :])
            print(f'{masked_words[k]} : {prob}\n vs. {most_likely} : {prob_ml}')
