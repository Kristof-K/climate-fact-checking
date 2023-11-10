from text_encoding.char_one_hot_encoder import CharOneHotEncoder


def get_encoder(config):
    # factory method that creates correct encoding object based on config file
    if config['word_based']:
        # return word encoder
        return None
    else:
        return CharOneHotEncoder(max_seq_length=config['max_seq_len'],
                                 max_output=config['max_output'],
                                 start_token=config['start_token'],
                                 stop_token=config['end_token'],
                                 mask_symbol=config['mask_symbol'])

