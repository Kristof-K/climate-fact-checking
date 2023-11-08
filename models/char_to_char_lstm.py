import numpy as np
import os
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.models import model_from_json

from models.masked_nlp_model import MaskedNLPModel
from text_encoding.char_one_hot_encoder import CharOneHotEncoder


class CharToCharLSTM(MaskedNLPModel):
    # based on Keras sequence-to-sequence model
    # https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
    # or as presented in: Lane, Hobson and Howard, Cole and Hapke, Hannes Max: Natural Language Processing in Action

    def __init__(self, text_encoder: CharOneHotEncoder, vocab_size: int, num_neurons: int,
                 batch_size: int, epochs: int, save_epochs: int, validation_split: float):
        self.text_encoder = text_encoder
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.save_epochs = save_epochs
        self.validation_split = validation_split

        # define Network structure
        # encoder encodes input sequence of arbitrary length (dimension None!)
        encoder_in = Input(shape=(None, self.vocab_size))
        encoder_out = LSTM(num_neurons, return_state=True)  # output hidden states
        # we do not return sequences, hence encoder_output == state_h; and state_c is the long-term memory state
        encoder_output, en_state_h, en_state_c = encoder_out(encoder_in)
        encoder_states = [en_state_h, en_state_c]

        # decoder gets the target sequence with start token as input and is trained to
        # reproduce the target sequence without start token
        decoder_in = Input(shape=(None, self.vocab_size))
        decoder_lstm = LSTM(num_neurons, return_sequences=True, return_state=True)
        decoder_intermed, de_state_h, de_state_c = decoder_lstm(decoder_in, initial_state=encoder_states)
        decoder_dense = Dense(self.vocab_size, activation='softmax')
        decoder_out = decoder_dense(decoder_intermed)

        self.model_combined = Model(inputs=[encoder_in, decoder_in], outputs=decoder_out)
        self.model_combined.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        # store the encoder and decoder part separately for inference
        self.encoder = Model(inputs=encoder_in, outputs=encoder_states)
        self.decoder = Model(inputs=[decoder_in, encoder_states],
                             outputs=[decoder_out, de_state_h, de_state_c])

    def train(self, samples_x: np.array, samples_y: np.array, samples_y_no_start: np.array,
              path: str):
        # save model structure
        model_struc_enc = self.encoder.to_json()
        with open(os.path.join(path, 'encoder.json'), 'w') as json_file:
            json_file.write(model_struc_enc)
        model_struc_enc = self.decoder.to_json()
        with open(os.path.join(path, 'decoder.json'), 'w') as json_file:
            json_file.write(model_struc_enc)

        i = 0
        loss_vals = dict([('loss', []), ('val_loss', [])])
        while i < self.epochs:
            history = self.model_combined.fit([samples_x, samples_y], samples_y_no_start,
                                              batch_size=self.batch_size, epochs=self.save_epochs,
                                              validation_split=self.validation_split)
            i += self.save_epochs
            loss_vals['loss'] += history.history['loss']
            loss_vals['val_loss'] += history.history['val_loss']

            self.encoder.save_weights(os.path.join(path, 'encoder_{}.h5'.format(i)))
            self.decoder.save_weights(os.path.join(path, 'decoder_{}.h5'.format(i)))
        MaskedNLPModel.print_loss(loss_vals, path)

    def load_model(self, path, epoch):
        with open(os.path.join(path, 'encoder.json'), 'r') as json_file:
            json_string = json_file.read()
        self.encoder = model_from_json(json_string)
        self.encoder.load_weights(os.path.join(path, f'encoder_{epoch}.h5'))

        with open(os.path.join(path, 'decoder.json'), 'r') as json_file:
            json_string = json_file.read()
        self.decoder = model_from_json(json_string)
        self.decoder.load_weights(os.path.join(path, f'decoder_{epoch}.h5'))

    def get_token_probability(self, x_num: np.array, masked_word: str):
        thought = self.encoder.predict(x_num, verbose=0)  # output + memory state
        target_seq = np.zeros((1, 1, self.vocab_size))
        # first element is start token
        target_seq[0, 0, self.text_encoder.char_to_index[self.text_encoder.start_token]] = 1.0

        prob = 1.0
        for char in masked_word + self.text_encoder.stop_token:
            char_probs, h, c = self.decoder.predict([target_seq] + thought, verbose=0)

            # multiply probabilities of corresponding characters in the masked word
            prob *= char_probs[0, -1, self.text_encoder.char_to_index[char]]

            # now look at next character
            target_seq = np.zeros((1, 1, self.vocab_size))
            target_seq[0, 0, self.text_encoder.char_to_index[char]] = 1.0
            thought = [h, c]
        return prob

    def get_most_likely_word(self, x_num: np.array):
        thought = self.encoder.predict(x_num, verbose=0)  # output + memory state
        target_seq = np.zeros((1, 1, self.vocab_size))
        # first element is start token
        target_seq[0, 0, self.text_encoder.char_to_index[self.text_encoder.start_token]] = 1.0
        word = ''
        new_char = self.text_encoder.start_token
        prob = 1.0

        while len(word) <= self.text_encoder.max_output:
            char_probs, h, c = self.decoder.predict([target_seq] + thought, verbose=0)

            # multiply probabilities of corresponding characters in the masked word
            max_index = np.argmax(char_probs[0, 0, :])
            new_char = self.text_encoder.index_to_char[max_index]
            prob *= char_probs[0, 0, max_index]

            if new_char == self.text_encoder.stop_token:
                break
            word += new_char

            # now look at next character
            target_seq = np.zeros((1, 1, self.vocab_size))
            target_seq[0, 0, self.text_encoder.char_to_index[new_char]] = 1.0
            thought = [h, c]
        return word, prob

