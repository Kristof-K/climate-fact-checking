from typing import Iterator, List
import numpy as np
import os
from keras.models import Model, model_from_json

from models.masked_nlp_model import MaskedNLPModel
from text_encoding.word2vec import MyWord2Vec


class MaskedWordModel(MaskedNLPModel):

    def __init__(self, model: Model, text_encoder: MyWord2Vec, batch_size: int, epochs: int,
                 save_epochs: int):
        self.model = model
        self.text_encoder = text_encoder

        self.vocab_size = text_encoder.get_vocab_size()
        self.batch_size = batch_size
        self.epochs = epochs
        self.save_epochs = save_epochs

    def train(self, generator: Iterator, steps: int, path: str):
        # save model structure
        model_struc_enc = self.model.to_json()
        with open(os.path.join(path, 'model.json'), 'w') as json_file:
            json_file.write(model_struc_enc)

        i = 0
        loss_vals = dict([('loss', [])])
        while i < self.epochs:
            history = self.model.fit(x=generator, batch_size=self.batch_size,
                                     epochs=self.save_epochs, steps_per_epoch=steps)
            i += self.save_epochs
            loss_vals['loss'] += history.history['loss']

            self.model.save_weights(os.path.join(path, 'model_{}.h5'.format(i)))
        MaskedNLPModel.print_loss(loss_vals, path)

    def load_model(self, path, epoch):
        with open(os.path.join(path, 'model.json'), 'r') as json_file:
            json_string = json_file.read()
        self.model = model_from_json(json_string)
        self.model.load_weights(os.path.join(path, f'model_{epoch}.h5'))

    def get_token_probabilities(self, x_num: np.array, masked_word: List[str]):
        probs = self.model.predict(x_num, verbose=0)

        return np.array(
            [probs[i, self.text_encoder.word_to_index[masked_word[i]]]
             if masked_word[i] in self.text_encoder.word_vectors
             else -1.0 for i in range(len(masked_word))]
        )

    def get_most_likely_words(self, x_num: np.array, n_beams: int = 5):
        # expect x_num has batch dimension 1
        probs = self.model.predict(x_num, verbose=0)
        k_largest = np.argsort(-1.0 * probs[0, :])[:n_beams]

        return self.text_encoder.decode(k_largest), probs[0, k_largest]
