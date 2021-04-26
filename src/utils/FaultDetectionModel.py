import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils import class_weight
from src.utils.data_utils.BotDataset import BotDataset


class FaultDetectionModel:
    def __init__(self, bot_datasets: list[BotDataset]):
        self.bot_datasets = bot_datasets
        self.model = self.build_model()

    @staticmethod
    def build_model() -> keras.Model:
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

        inputs = keras.Input(shape=(9, 10))
        x = layers.Conv1D(filters=16,
                          kernel_size=3,
                          strides=1,
                          padding='same',
                          activation='relu',
                          )(inputs)
        x = layers.Conv1D(filters=16,
                          kernel_size=3,
                          strides=1,
                          padding='same',
                          activation='relu',
                          )(x)
        x = layers.MaxPooling1D(pool_size=2,
                                padding='valid')(x)
        x = layers.Conv1D(filters=32,
                          kernel_size=2,
                          strides=1,
                          padding='same',
                          activation='relu',
                          )(x)
        x = layers.Conv1D(filters=32,
                          kernel_size=2,
                          strides=1,
                          padding='same',
                          activation='relu',
                          )(x)
        x = layers.MaxPooling1D(pool_size=2,
                                padding='valid')(x)
        x = layers.Conv1D(filters=64,
                          kernel_size=2,
                          strides=1,
                          padding='same',
                          activation='relu',
                          )(x)
        x = layers.Conv1D(filters=64,
                          kernel_size=2,
                          strides=1,
                          padding='same',
                          activation='relu',
                          )(x)
        x = layers.MaxPooling1D(pool_size=2,
                                padding='valid')(x)
        x = layers.LSTM(3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        model = keras.Model(inputs, outputs)
        model.summary()
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                      loss=keras.losses.BinaryCrossentropy(),
                      metrics=metrics)

        return model

    def train_model(self):
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, min_delta=0.05)

        for bot in range(len(self.bot_datasets)):
            print('bot: ' + str(bot) + ' Training')

            if any(self.bot_datasets[bot].target_train_dataset):
                numpy_class_weights = class_weight.compute_class_weight(
                    'balanced',
                    np.unique(
                        self.bot_datasets[bot].target_train_dataset),
                    self.bot_datasets[bot].target_train_dataset)
                class_weights = dict(enumerate(numpy_class_weights))
            else:
                class_weights = {0: 1.,
                                 1: 1.}

            train_ds = tf.data.Dataset.from_tensor_slices(
                    (np.expand_dims(self.bot_datasets[bot].train_dataset, 1),
                     np.expand_dims(self.bot_datasets[bot].target_train_dataset, -1))
                )

            validation_ds = tf.data.Dataset.from_tensor_slices(
                (np.expand_dims(self.bot_datasets[bot].validation_dataset, 1),
                 np.expand_dims(self.bot_datasets[bot].target_validation_dataset, -1))
            )

            self.model.fit(train_ds,
                           epochs=3,
                           batch_size=10,
                           callbacks=[callback],
                           validation_data=validation_ds,
                           class_weight=class_weights,
                           shuffle=False,
                           verbose=1)

            print('bot: ' + str(bot) + ' Testing')
            test_ds = tf.data.Dataset.from_tensor_slices(
                (np.expand_dims(self.bot_datasets[bot].test_dataset, 1),
                 np.expand_dims(self.bot_datasets[bot].target_test_dataset, -1))
            )
            # returns loss and metrics
            loss, tp, fp, tn, fn, accuracy, precision, recall, auc, prc = self.model.evaluate(test_ds)
            output = "loss: %.2f" % loss + " tp: %.2f" % tp + " fp: %.2f" % fp + " tn: %.2f" % tn + " fn: %.2f" % fn
            output += " accuracy: %.2f" % accuracy + " precision: %.2f" % precision + " recall: %.2f" % recall
            output += " auc: %.2f" % auc + " prc: %.2f" % prc
            print(output)
