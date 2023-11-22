from typing import List

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Bidirectional
from keras.optimizers import RMSprop

from models.masked_char_nlp_model import MaskedChartoChar
from text_encoding.char_one_hot_encoder import CharOneHotEncoder


class CharToCharBiLSTM(MaskedChartoChar):
    # based on Keras sequence-to-sequence model but using bidirectional LSTM layer
    # https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
    # or as presented in: Lane, Hobson and Howard, Cole and Hapke, Hannes Max: Natural Language Processing in Action

    def __init__(self, text_encoder: CharOneHotEncoder, num_neurons: List[int],
                 batch_size: int, epochs: int, save_epochs: int, learning_rate: float):
        vocab_size = text_encoder.get_vocab_size()
        # define Network structure
        # encoder encodes input sequence of arbitrary length (dimension None!)
        encoder_in = Input(shape=(None, vocab_size))

        encoder_lstms = []
        for i in range(1, len(num_neurons)):
            encoder_lstms.append(Bidirectional(LSTM(num_neurons[i - 1], return_sequences=True)))
        encoder_lstms.append(Bidirectional(LSTM(num_neurons[-1], return_state=True)))  # output hidden states

        encoder_out = encoder_in
        for lstm_l in encoder_lstms:
            encoder_out = lstm_l(encoder_out)       # send input through the layers

        # we do not return sequences, hence encoder_output == state_h; and state_c is the long-term memory state
        encoder_output, en_state_h, en_state_c, en_state_h_back, en_state_c_back = encoder_out
        # sum states as decoder is a unidirectional LSTM, so concatenating would double the dimension
        encoder_states = [en_state_h + en_state_h_back, en_state_c + en_state_c_back]

        # decoder gets the target sequence with start token as input and is trained to
        # reproduce the target sequence without start token
        decoder_in = Input(shape=(None, vocab_size))

        decoder_lstms = []
        for i in reversed(range(1, len(num_neurons))):
            decoder_lstms.append(LSTM(num_neurons[i], return_sequences=True))
        decoder_lstms.append(LSTM(num_neurons[0], return_sequences=True, return_state=True))  # output hidden states

        decoder_proc = decoder_lstms[0](decoder_in, initial_state=encoder_states)      # initially we need encoder state
        for i in range(1, len(decoder_lstms)):
            decoder_proc = decoder_lstms[i](decoder_proc)       # send input through the layers

        decoder_intermed, de_state_h, de_state_c = decoder_proc
        decoder_dense = Dense(vocab_size, activation='softmax')
        decoder_out = decoder_dense(decoder_intermed)

        model_combined = Model(inputs=[encoder_in, decoder_in], outputs=decoder_out)
        optimizer = RMSprop(learning_rate=learning_rate)
        model_combined.compile(optimizer=optimizer, loss='categorical_crossentropy')
        # store the encoder and decoder part separately for inference
        encoder = Model(inputs=encoder_in, outputs=encoder_states)
        decoder = Model(inputs=[decoder_in] + encoder_states,
                        outputs=[decoder_out, de_state_h, de_state_c])

        super().__init__(model_combined, encoder, decoder, text_encoder,
                         batch_size=batch_size, epochs=epochs, save_epochs=save_epochs)
