from text_encoding.text_encoder import TextEncoder

# character models
from models.char_to_char_lstm import CharToCharLSTM
from models.char_to_char_bi_lstm import CharToCharBiLSTM

# word models
from models.wordvec_to_word_lstm import WordvecToWordLSTM
from models.wordvec_to_word_bi_lstm import WordvecToWordBiLSTM


def get_model(text_encoder: TextEncoder, config: dict):
    if config['model'] == 'CharToCharLSTM':
        return CharToCharLSTM(text_encoder, num_neurons=config['num_neurons'],
                              batch_size=config['batch_size'], epochs=config['epochs'],
                              save_epochs=config['save_epochs'], learning_rate=config["learning_rate"])
    elif config['model'] == 'CharToCharBiLSTM':
        return CharToCharBiLSTM(text_encoder, num_neurons=config['num_neurons'],
                                batch_size=config['batch_size'], epochs=config['epochs'],
                                save_epochs=config['save_epochs'], learning_rate=config["learning_rate"])
    elif config['model'] == 'WordvecToWordLSTM':
        return WordvecToWordLSTM(text_encoder, num_neurons=config['num_neurons'],
                                 batch_size=config['batch_size'], epochs=config['epochs'],
                                 save_epochs=config['save_epochs'], learning_rate=config["learning_rate"])
    elif config['model'] == 'WordvecToWordBiLSTM':
        return WordvecToWordBiLSTM(text_encoder, num_neurons=config['num_neurons'],
                                   batch_size=config['batch_size'], epochs=config['epochs'],
                                   save_epochs=config['save_epochs'], learning_rate=config["learning_rate"])
