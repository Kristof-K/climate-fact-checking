from typing import Iterator
import numpy as np
import os
from keras.models import Model, model_from_json

from models.masked_nlp_model import MaskedNLPModel
from text_encoding.char_one_hot_encoder import CharOneHotEncoder


class MaskedChartoChar(MaskedNLPModel):
    # class that defines functionality for char to char models, i.e. specific models must just provide
    # the encoder and decoder element and the joint model for training

    def __init__(self, model: Model, encoder: Model, decoder: Model, text_encoder: CharOneHotEncoder,
                 batch_size: int, epochs: int, save_epochs: int):
        self.model = model
        self.encoder = encoder
        self.decoder = decoder
        self.text_encoder = text_encoder

        self.vocab_size = text_encoder.get_vocab_size()
        self.batch_size = batch_size
        self.epochs = epochs
        self.save_epochs = save_epochs

    @staticmethod
    def _wrap_generator(data_generator: Iterator):
        for x_num, y_num in data_generator:
            # our actual learning targets are the shifted y-values as we are using them also as input
            y_num_no_start = np.zeros_like(y_num)
            y_num_no_start[:, :-1, :] = y_num[:, 1:, :]     # just forget the start token

            yield (x_num, y_num), y_num_no_start

    def train(self, generator: Iterator, steps: int, path: str):
        # either generator and steps should be given or all three samples_*

        # save model structure
        model_struc_enc = self.encoder.to_json()
        with open(os.path.join(path, 'encoder.json'), 'w') as json_file:
            json_file.write(model_struc_enc)
        model_struc_enc = self.decoder.to_json()
        with open(os.path.join(path, 'decoder.json'), 'w') as json_file:
            json_file.write(model_struc_enc)

        i = 0
        loss_vals = dict([('loss', [])])
        while i < self.epochs:
            history = self.model.fit(x=MaskedChartoChar._wrap_generator(generator),
                                     batch_size=self.batch_size, epochs=self.save_epochs, steps_per_epoch=steps)
            i += self.save_epochs
            loss_vals['loss'] += history.history['loss']

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
        # first element is start token
        target_seq = self.text_encoder.encode_one_y(self.text_encoder.start_token)

        prob = 1.0
        for char in masked_word + self.text_encoder.stop_token:
            char_probs, h, c = self.decoder.predict([target_seq] + thought, verbose=0)

            # multiply probabilities of corresponding characters in the masked word
            prob *= char_probs[0, -1, self.text_encoder.char_to_index[char]]

            # now look at next character
            target_seq = self.text_encoder.encode_one_y(char)
            thought = [h, c]
        return prob

    def get_most_likely_words(self, x_num: np.array, n_beams: int = 5):
        # use beam search to look for the approximately most likely words
        thought = self.encoder.predict(x_num, verbose=0)  # output + memory state
        # first element is start token
        target_seq = self.text_encoder.encode_one_y(self.text_encoder.start_token)

        prev_chars, prev_probs, prev_thoughts = self._get_next_k_char_prob(target_seq, thought, n_beams)
        all_finished = np.array([char == self.text_encoder.stop_token for char in prev_chars])
        indices = np.arange(len(all_finished))
        prev_words = prev_chars

        while not np.all(all_finished):
            col_chars = [prev_chars[i] for i in indices[all_finished]]
            col_probs = prev_probs[indices[all_finished]]
            col_thoughts = [prev_thoughts[i] for i in indices[all_finished]]
            col_words = [prev_words[i] for i in indices[all_finished]]

            # get the next candidates
            for i in indices[np.bitwise_not(all_finished)]:
                add_chars, mult_probs, add_thoughts = self._get_next_k_char_prob(
                    self.text_encoder.encode_one_y(prev_chars[i]), prev_thoughts[i], n_beams
                )
                col_chars += add_chars
                col_words += [prev_words[i] + char if char != self.text_encoder.stop_token else prev_words[i]
                              for char in add_chars]
                col_probs = np.hstack((col_probs, prev_probs[i] * mult_probs))
                col_thoughts += add_thoughts

            # filter for the k best candidates
            k_largest = np.argsort(-1.0 * col_probs)[:n_beams]

            prev_chars = [col_chars[s] for s in k_largest]
            prev_words = [col_words[s] for s in k_largest]
            prev_probs = col_probs[k_largest]
            prev_thoughts = [col_thoughts[s] for s in k_largest]
            all_finished = np.bitwise_or(
                np.array([char == self.text_encoder.stop_token for char in prev_chars]),
                np.array([len(word) >= self.text_encoder.max_output for word in prev_words])
            )

        order = np.argsort(-1.0 * prev_probs)      # return words sorted by likelihood
        return [prev_words[s] for s in order], prev_probs[order]

    def _get_next_k_char_prob(self, target_seq, thought, k):
        char_probs, h, c = self.decoder.predict([target_seq] + thought, verbose=0)

        k_largest = np.argsort(-1.0 * char_probs[0, 0, :])[:k]

        new_chars = [self.text_encoder.index_to_char[i] for i in k_largest]
        probs = char_probs[0, 0, k_largest]
        new_thoughts = [[h, c]] * k

        return new_chars, probs, new_thoughts
