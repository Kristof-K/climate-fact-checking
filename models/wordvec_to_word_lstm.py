from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.optimizers import RMSprop

from models.masked_word_nlp_model import MaskedWordModel
from text_encoding.word2vec import MyWord2Vec


class WordvecToWordLSTM(MaskedWordModel):
    # Use LSTM layers to process sentences encoded by word embeddings, and use Dense softmax
    # learn distribution over masked words

    def __init__(self, text_encoder: MyWord2Vec, num_neurons: int,
                 batch_size: int, epochs: int, save_epochs: int, learning_rate: float):
        # define Network structure
        # encoder encodes input sequence of arbitrary length (dimension None!)
        model_in = Input(shape=(None, text_encoder.vec_dim))
        # we cannot output sequences (arbitrary length!) as we want to process input further
        model_lstm = LSTM(num_neurons, return_sequences=False)
        # just process the intermediate state to make the prediction
        model_dense = Dense(text_encoder.get_vocab_size(), activation='softmax')
        model_out = model_dense(model_lstm(model_in))

        model = Model(inputs=model_in, outputs=model_out)
        optimizer = RMSprop(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy')

        super().__init__(model, text_encoder, batch_size=batch_size, epochs=epochs, save_epochs=save_epochs)
