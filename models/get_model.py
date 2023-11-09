from models.char_to_char_lstm import CharToCharLSTM
from models.char_to_char_bi_lstm import CharToCharBiLSTM
from text_encoding.text_encoder import TextEncoder


def get_model(text_encoder: TextEncoder, vocab_size: int, config: dict):
    match config['model']:
        case 'CharToCharLSTM':
            return CharToCharLSTM(text_encoder, vocab_size, num_neurons=config['num_neurons'],
                                  batch_size=config['batch_size'], epochs=config['epochs'],
                                  save_epochs=config['save_epochs'], validation_split=config['validation_split'],
                                  learning_rate=config["learning_rate"])
        case 'CharToCharBiLSTM':
            return CharToCharBiLSTM(text_encoder, vocab_size, num_neurons=config['num_neurons'],
                                  batch_size=config['batch_size'], epochs=config['epochs'],
                                  save_epochs=config['save_epochs'], validation_split=config['validation_split'],
                                  learning_rate=config["learning_rate"])
