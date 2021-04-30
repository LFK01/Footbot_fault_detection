import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils import class_weight
from src.utils.Parser import Parser
from src.utils.data_utils.BotDataset import BotDataset


class ConvLstmModel:
    def __init__(self, bot_datasets: list[BotDataset]):
        self.LSTM_length = Parser.read_lstm_length()
        self.time_window = Parser.read_time_window()
        self.bot_datasets = bot_datasets
        self.model = self.build_model()

    @staticmethod
    def build_model() -> keras.Model:

        time_window = Parser.read_time_window()

        cnn = tf.keras.Sequential()

        cnn.add(layers.Conv2D(filters=16,
                              kernel_size=(3, 3),
                              strides=1,
                              padding='same',
                              activation='relu',
                              name='First_Conv2D_1'
                              ))
        cnn.add(layers.Conv2D(filters=16,
                              kernel_size=(3, 3),
                              strides=1,
                              padding='same',
                              activation='relu',
                              name='First_Conv2D_2'
                              ))
        cnn.add(layers.MaxPooling2D(pool_size=2,
                                    padding='valid',
                                    name='First_MaxPool'))
        cnn.add(layers.Conv2D(filters=32,
                              kernel_size=(3, 3),
                              strides=1,
                              padding='same',
                              activation='relu',
                              name='Second_Conv2D_1'
                              ))
        cnn.add(layers.Conv2D(filters=32,
                              kernel_size=(3, 3),
                              strides=1,
                              padding='same',
                              activation='relu',
                              name='Second_Conv2D_2'
                              ))
        cnn.add(layers.MaxPooling2D(pool_size=2,
                                    padding='valid',
                                    name='Second_MaxPool'))
        cnn.add(layers.Conv2D(filters=64,
                              kernel_size=(2, 2),
                              strides=1,
                              padding='same',
                              activation='relu',
                              name='Third_Conv2D_1'
                              ))
        cnn.add(layers.Conv2D(filters=64,
                              kernel_size=(2, 2),
                              strides=1,
                              padding='same',
                              activation='relu',
                              name='Third_Conv2D_2'
                              ))
        cnn.add(layers.MaxPooling2D(pool_size=2,
                                    padding='valid',
                                    name='Third_MaxPool'))
        cnn.add(layers.Flatten(name='Flatten'))
        cnn.build(input_shape=(None, 9, 10, 1))
        cnn.summary()
        for layer in cnn.layers:
            print('Layer: ' + layer.name + ' Input shape: ' + str(layer.input_shape))
            print('Output shape: ' + str(layer.output_shape))

        rnn = tf.keras.Sequential(name='RNN')
        rnn.add(layers.LSTM(40, return_sequences=True, name='LSTM'))
        rnn.build(input_shape=(None, 10, 64))
        rnn.summary()
        for layer in rnn.layers:
            print('Layer: ' + layer.name + ' Input shape: ' + str(layer.input_shape))
            print('Output shape: ' + str(layer.output_shape))

        dense = tf.keras.Sequential(name='Dense')
        dense.add(layers.Dense(64, activation='relu', name='Dense64'))
        dense.add(layers.Dense(1, activation='sigmoid', name='Dense1'))
        dense.build(input_shape=(None, 10, 40))
        dense.summary()
        for layer in dense.layers:
            print('Layer: ' + layer.name + ' Input shape: ' + str(layer.input_shape))
            print('Output shape: ' + str(layer.output_shape))

        final = tf.keras.Sequential()

        final.add(layers.TimeDistributed(cnn,
                                         input_shape=(10, 9, time_window, 1),
                                         name='TimeDistributed'))
        final.add(rnn)
        final.add(dense)
        final.build(input_shape=(10, 9, time_window, 1))
        final.summary()
        for layer in final.layers:
            print('Layer: ' + layer.name + ' Input shape: ' + str(layer.input_shape))
            print('Output shape: ' + str(layer.output_shape))

        metrics = [
            keras.metrics.TruePositives(name='tp'),
            keras.metrics.FalsePositives(name='fp'),
            keras.metrics.TrueNegatives(name='tn'),
            keras.metrics.FalseNegatives(name='fn'),
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
            keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
        ]

        final.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                      loss=keras.losses.BinaryCrossentropy(),
                      metrics=metrics)

        return final

    def train_model(self):
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, min_delta=0.05)

        batch_size = 16

        for bot in range(len(self.bot_datasets)):
            print('bot: ' + str(bot) + ' Training')

            if any(self.bot_datasets[bot].target_train_dataset):
                numpy_class_weights = class_weight.compute_class_weight(
                    'balanced',
                    np.unique(self.bot_datasets[bot].target_train_dataset),
                    self.bot_datasets[bot].target_train_dataset
                )
                class_weights = dict(enumerate(numpy_class_weights))
            else:
                class_weights = {0: 1.,
                                 1: 1.}

            train_ds, validation_ds, test_ds = self.datasets_preparation(bot, batch_size)

            train_samples = train_ds.cardinality()
            train_steps = int(np.ceil(train_samples / batch_size))
            validation_samples = validation_ds.cardinality()
            validation_steps = int(np.ceil(validation_samples / batch_size))

            self.model.fit(train_ds,
                           epochs=10,
                           batch_size=2048,
                           callbacks=[callback],
                           validation_data=validation_ds,
                           shuffle=False,
                           verbose=1)

            print('bot: ' + str(bot) + ' Testing')

            test_sample = test_ds.element_spec[0].shape[1]
            test_step_per_epoch = int(np.ceil(test_sample / batch_size))
            # returns loss and metrics
            loss, tp, fp, tn, fn, accuracy, precision, recall, auc, prc = self.model.evaluate(test_ds,
                                                                                              batch_size=batch_size,
                                                                                              steps_per_epoch=
                                                                                              test_step_per_epoch)
            output = "loss: %.2f" % loss + " tp: %.2f" % tp + " fp: %.2f" % fp + " tn: %.2f" % tn + " fn: %.2f" % fn
            output += " accuracy: %.2f" % accuracy + " precision: %.2f" % precision + " recall: %.2f" % recall
            output += " auc: %.2f" % auc + " prc: %.2f" % prc
            print(output)

    def datasets_preparation(self, bot: int, batch_size: int):
        batch_train_array, batch_target_train_array = self.add_batch_dimension(
            numpy_array=self.bot_datasets[bot].train_dataset,
            target_numpy_array=self.bot_datasets[bot].target_train_dataset
        )

        # batch_train_array = self.bot_datasets[bot].train_dataset
        # batch_target_train_array = self.bot_datasets[bot].target_train_dataset

        train_ds = self.build_features_target_dataset(batch_size=batch_size,
                                                      batch_array=batch_train_array,
                                                      batch_target_array=batch_target_train_array)

        batch_validation_array, batch_target_validation_array = self.add_batch_dimension(
            numpy_array=self.bot_datasets[bot].validation_dataset,
            target_numpy_array=self.bot_datasets[bot].target_validation_dataset
        )

        # batch_validation_array = self.bot_datasets[bot].validation_dataset
        # batch_target_validation_array = self.bot_datasets[bot].validation_dataset

        validation_ds = self.build_features_target_dataset(batch_size=batch_size,
                                                           batch_array=batch_validation_array,
                                                           batch_target_array=batch_target_validation_array)

        batch_test_array, batch_target_test_array = self.add_batch_dimension(
            numpy_array=self.bot_datasets[bot].test_dataset,
            target_numpy_array=self.bot_datasets[bot].target_test_dataset
        )

        # batch_test_array = self.bot_datasets[bot].test_dataset
        # batch_target_test_array = self.bot_datasets[bot].target_test_dataset

        test_ds = self.build_features_target_dataset(batch_size=batch_size,
                                                     batch_array=batch_test_array,
                                                     batch_target_array=batch_target_test_array)

        return train_ds, validation_ds, test_ds

    def add_batch_dimension(self, numpy_array: np.ndarray, target_numpy_array: np.ndarray):
        time_batch_numpy_array = np.asarray(
            np.stack([numpy_array[i:i + self.time_window] for i in
                      range(len(numpy_array) - self.time_window)]
                     )
        )

        time_batch_target_numpy_array = np.asarray(
            np.stack([target_numpy_array[i:i + self.time_window] for i in
                      range(len(target_numpy_array) - self.time_window)])
        )

        return time_batch_numpy_array, time_batch_target_numpy_array

    @staticmethod
    def build_features_target_dataset(batch_size: int,
                                      batch_array: np.ndarray,
                                      batch_target_array: np.ndarray):

        ds_from_tensor = tf.data.Dataset.from_tensors(
            (np.expand_dims(batch_array, -1),
             np.expand_dims(batch_target_array, -1))
        )

        ds_from_slices = tf.data.Dataset.from_tensor_slices(
            (np.expand_dims(batch_array, (1, -1)),
             np.expand_dims(batch_target_array, (1, -1)))
        )

        ds_from_tensor = ds_from_tensor.prefetch(tf.data.AUTOTUNE)

        ds_from_slices = ds_from_slices.prefetch(tf.data.AUTOTUNE)

        return ds_from_slices
