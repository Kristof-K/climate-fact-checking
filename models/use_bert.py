from typing import Iterator
import numpy as np
import os
from keras.models import model_from_json
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from transformers import TFAutoModelForMaskedLM, TFDistilBertForMaskedLM

from models.masked_nlp_model import MaskedNLPModel
from text_encoding.bert_tokenizers import MyBertTokenizer


class WrappedBert(MaskedNLPModel):
    # compare https://huggingface.co/bert-base-uncased
    # compare https://huggingface.co/learn/nlp-course/chapter7/3?fw=tf
    # compare https://www.analyticsvidhya.com/blog/2022/09/fine-tuning-bert-with-masked-language-modeling/

    # https://huggingface.co/docs/transformers/main_classes/output
    # BERT MaskedLM outputs energy values (before softmax is applied)

    # https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter7/section3_pt.ipynb
    # https://www.analyticsvidhya.com/blog/2022/09/fine-tuning-bert-with-masked-language-modeling/
    # https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/nlp/ipynb/masked_language_modeling.ipynb#scrollTo=HSmgQ_pyy8_G

    def __init__(self, model_name: str, text_encoder: MyBertTokenizer, batch_size: int, epochs: int, save_epochs: int,
                 learning_rate: float):
        self.text_encoder = text_encoder
        self.model = TFAutoModelForMaskedLM.from_pretrained(model_name)

        self.vocab_size = text_encoder.get_vocab_size()
        self.batch_size = batch_size
        self.epochs = epochs
        self.save_epochs = save_epochs
        self.learning_rate = learning_rate

    def _wrap_generator(self, data_generator: Iterator):
        for x_num, y_num in data_generator:
            # we need to add attention values to indicate which values have been padded
            attention = 1 * (x_num != self.text_encoder.pad_token)
            yield (x_num, attention), y_num

    def train(self, generator: Iterator, steps: int, path: str):
        optimizer = Adam(learning_rate=self.learning_rate)
        loss_fn = SparseCategoricalCrossentropy(from_logits=True)
        self.model.compile(optimizer=optimizer, loss=loss_fn)

        # save model structure: no need as we just reinitialize the model

        i = 0
        loss_vals = dict([('loss', [])])
        while i < self.epochs:
            history = self.model.fit(x=self._wrap_generator(generator), batch_size=self.batch_size,
                                     epochs=self.save_epochs, steps_per_epoch=steps)
            i += self.save_epochs
            loss_vals['loss'] += history.history['loss']

            self.model.save_weights(os.path.join(path, 'model_{}.h5'.format(i)))
        MaskedNLPModel.print_loss(loss_vals, path)

    def load_model(self, path, epoch):
        self.model.load_weights(os.path.join(path, f'model_{epoch}.h5'))

    def get_token_probability(self, x_num: np.array, masked_word: str):
        scores = self.model(x_num).logits.numpy()
        masked_position = np.argwhere(x_num[0, :] == self.text_encoder.mask_token_id)[0, 0]

        true_word = self.text_encoder.encode_one_y(masked_word)

        return scores[0, masked_position, true_word]

    def get_most_likely_words(self, x_num: np.array, n_beams: int = 5):
        scores = self.model(x_num).logits.numpy()
        masked_position = np.argwhere(x_num[0, :] == self.text_encoder.mask_token_id)[0, 0]
        k_largest = np.argsort(-1.0 * scores[0, masked_position, :])[:n_beams]

        # text encoder returns tokens as sentence separated by spaces
        return self.text_encoder.decode(k_largest), scores[0, masked_position, k_largest]
