from typing import List

from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.optimizers import RMSprop
from keras.losses import CategoricalCrossentropy

from models.masked_word_nlp_model import MaskedWordModel
from text_encoding.word2vec import MyWord2Vec


class WordvecToWordLSTM(MaskedWordModel):
    # Use LSTM layers to process sentences encoded by word embeddings, and use Dense softmax
    # learn distribution over masked words

    def __init__(self, text_encoder: MyWord2Vec, num_neurons: List[int],
                 batch_size: int, epochs: int, save_epochs: int, learning_rate: float):
        # define Network structure
        # encoder encodes input sequence of arbitrary length (dimension None!)
        model_in = Input(shape=(None, text_encoder.vec_dim))

        lstm_layers = []
        for i in range(1, len(num_neurons)):
            lstm_layers.append(LSTM(num_neurons[i - 1], return_sequences=True))
        # we cannot output sequences (arbitrary length!) as we want to process input further
        lstm_layers.append(LSTM(num_neurons[-1], return_sequences=False))
        # just process the intermediate state to make the prediction
        model_dense = Dense(text_encoder.get_vocab_size(), activation='softmax')

        model_out = model_in
        for lstm_l in lstm_layers:
            model_out = lstm_l(model_out)
        model_out = model_dense(model_out)

        model = Model(inputs=model_in, outputs=model_out)
        optimizer = RMSprop(learning_rate=learning_rate)
        loss_fn = CategoricalCrossentropy(from_logits=True)
        model.compile(optimizer=optimizer, loss=loss_fn)

        super().__init__(model, text_encoder, batch_size=batch_size, epochs=epochs, save_epochs=save_epochs)
