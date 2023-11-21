from text_encoding.text_encoder import TextEncoder

# character models
from models.char_to_char_lstm import CharToCharLSTM
from models.char_to_char_bi_lstm import CharToCharBiLSTM

# word models
from models.wordvec_to_word_lstm import WordvecToWordLSTM
from models.wordvec_to_word_bi_lstm import WordvecToWordBiLSTM
from models.use_bert import WrappedBert

C_2_C_LSTM = 'CharToCharLSTM'
C_2_C_BILSTM = 'CharToCharBiLSTM'
WV_2_W_LSTM = 'WordvecToWordLSTM'
WV_2_W_BILSTM = 'WordvecToWordBiLSTM'
BERT = 'bert-base-uncased'
DISTILL_BERT = 'distilbert-base-uncased'


def get_model(text_encoder: TextEncoder, config: dict):
    if config['model'] == C_2_C_LSTM:
        return CharToCharLSTM(text_encoder, num_neurons=config['num_neurons'],
                              batch_size=config['batch_size'], epochs=config['epochs'],
                              save_epochs=config['save_epochs'], learning_rate=config["learning_rate"])
    elif config['model'] == C_2_C_BILSTM:
        return CharToCharBiLSTM(text_encoder, num_neurons=config['num_neurons'],
                                batch_size=config['batch_size'], epochs=config['epochs'],
                                save_epochs=config['save_epochs'], learning_rate=config["learning_rate"])
    elif config['model'] == WV_2_W_LSTM:
        return WordvecToWordLSTM(text_encoder, num_neurons=config['num_neurons'],
                                 batch_size=config['batch_size'], epochs=config['epochs'],
                                 save_epochs=config['save_epochs'], learning_rate=config["learning_rate"])
    elif config['model'] == WV_2_W_BILSTM:
        return WordvecToWordBiLSTM(text_encoder, num_neurons=config['num_neurons'],
                                   batch_size=config['batch_size'], epochs=config['epochs'],
                                   save_epochs=config['save_epochs'], learning_rate=config["learning_rate"])
    elif config['model'] == BERT:
        return WrappedBert('bert-base-uncased', text_encoder, batch_size=config['batch_size'],
                           epochs=config['epochs'], save_epochs=config['save_epochs'],
                           learning_rate=config["learning_rate"])
    elif config['model'] == DISTILL_BERT:
        return WrappedBert('distilbert-base-uncased', text_encoder, batch_size=config['batch_size'],
                           epochs=config['epochs'], save_epochs=config['save_epochs'],
                           learning_rate=config["learning_rate"])
