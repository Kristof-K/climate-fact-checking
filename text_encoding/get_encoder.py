from text_encoding.char_one_hot_encoder import CharOneHotEncoder
from text_encoding.word2vec import MyWord2Vec
from text_encoding.bert_tokenizers import MyBertTokenizer
from models.get_model import BERT, DISTILL_BERT


def get_encoder(config):
    # factory method that creates correct encoding object based on config file
    if config['word_based']:
        # return word encoder
        if config['model'] == BERT or config['model'] == DISTILL_BERT:
            return MyBertTokenizer(model_name=config['model'],
                                   max_seq_length=config['max_seq_len'])
        else:
            return MyWord2Vec(max_seq_length=config['max_seq_len'],
                              max_output=config['max_output'],
                              mask_symbol=config['mask_symbol'],
                              data_file=config['data_file'])
    else:
        return CharOneHotEncoder(max_seq_length=config['max_seq_len'],
                                 max_output=config['max_output'],
                                 start_token=config['start_token'],
                                 stop_token=config['end_token'],
                                 mask_symbol=config['mask_symbol'])
