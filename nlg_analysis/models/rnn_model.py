import string
from typing import List

import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

from nlg_analysis.cfg import TrainConfig
from nlg_analysis.models.base_model import BaseModel
from nlg_analysis.models.gru_model import GRUModel
from nlg_analysis.models.one_step_model import OneStepModel


class RNNModel(BaseModel):
    def __init__(
            self,
            path2model: str = None,
            embedding_dim: int = 256,
            rnn_units: int = 1024,
    ):
        self.ids_from_chars = preprocessing.StringLookup(
            vocabulary=list(string.ascii_lowercase) + list(string.ascii_uppercase) + list("ęóąśłżźćńĘÓĄŚŁŻŹĆŃ") + ['\n',
                                                                                                                   ' ',
                                                                                                                   '!',
                                                                                                                   "'",
                                                                                                                   ',',
                                                                                                                   '.',
                                                                                                                   '?'],
            mask_token=None
        )
        self.chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=self.ids_from_chars.get_vocabulary(),
            invert=True,
            mask_token=None
        )
        self.model = GRUModel(
            vocab_size=len(self.ids_from_chars.get_vocabulary()),
            embedding_dim=embedding_dim,
            rnn_units=rnn_units
        )
        loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.model.compile(optimizer='adam', loss=loss)
        if path2model is not None:
            self.model.load_weights(path2model)
            self.one_step_model = OneStepModel(self.model, self.chars_from_ids, self.ids_from_chars)

    def generate_transcript(self, questions: List) -> str:
        trans_txt = ""
        states = None
        for q in questions:
            trans_txt = trans_txt + q + "\n"
            next_char = tf.constant([trans_txt])
            result = [trans_txt]
            for n in range(1000):
                next_char, states = self.one_step_model.generate_one_step(next_char, states=states)
                result.append(next_char)
            result = tf.strings.join(result)
            answer = result[0].numpy().decode('utf-8')
            answer = answer[len(trans_txt):]
            answer = answer.split("\n")[0]
            trans_txt = trans_txt + answer + "\n"
        return trans_txt

    @staticmethod
    def approach() -> str:
        return "RNN_MODEL"

    def train(self, output_path: str, train_config: TrainConfig):
        text = open(train_config.train_ds_path, 'rb').read().decode(encoding='utf-8')
        all_ids = self.ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
        ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
        sequences = ids_dataset.batch(train_config.seq_length + 1, drop_remainder=True)
        dataset = sequences.map(self.split_input_target)

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=output_path,
            save_weights_only=True)

        dataset = (
            dataset
            .shuffle(train_config.buffer_size)
            .batch(train_config.batch_size, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE))
        self.model.fit(dataset, epochs=train_config.epochs, callbacks=[checkpoint_callback])

    @staticmethod
    def split_input_target(sequence):
        input_text = sequence[:-1]
        target_text = sequence[1:]
        return input_text, target_text
