from text_encoding.char_one_hot_encoder import CharOneHotEncoder
from text_encoding.word2vec import MyWord2Vec


def get_encoder(config):
    # factory method that creates correct encoding object based on config file
    if config['word_based']:
        # return word encoder
        return MyWord2Vec(max_seq_length=config['max_seq_len'],
                          max_output=config['max_output'],
                          mask_symbol=config['mask_symbol'])
    else:
        return CharOneHotEncoder(max_seq_length=config['max_seq_len'],
                                 max_output=config['max_output'],
                                 start_token=config['start_token'],
                                 stop_token=config['end_token'],
                                 mask_symbol=config['mask_symbol'])

