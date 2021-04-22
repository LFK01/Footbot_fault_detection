import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.utils.DataWizard import DataWizard


class FaultDetectionModel:
    def __init__(self, data_wizard: DataWizard):
        self.data_wizard = data_wizard
        self.model = self.build_model()

    def build_model(self) -> keras.Model:
        inputs = keras.Input(shape=(9, 10))
        x = layers.Flatten()(inputs)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(128, activation="relu")(x)
        outputs = layers.Dense(1, activation="softmax")(x)
        model = keras.Model(inputs, outputs)
        model.summary()
        model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
                      loss=keras.losses.CategoricalCrossentropy(),
                      metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")])
        return model

    def train_model(self):
        for bot in range(self.data_wizard.train_ds.shape[0]):
            print('bot: ' + str(bot) + ' Training')
            for experiment in range(self.data_wizard.train_ds.shape[1]):

                train_ds = tf.data.Dataset.from_tensor_slices(
                    (np.expand_dims(self.data_wizard.train_ds[bot, experiment], 1),
                     np.expand_dims(self.data_wizard.train_target_ds[bot, experiment], -1))
                )

                validation_experiments_number = self.data_wizard.validation_ds.shape[1]
                validation_experiment = experiment % validation_experiments_number
                validation_ds = tf.data.Dataset.from_tensor_slices(
                    (np.expand_dims(self.data_wizard.validation_ds[bot, validation_experiment], 1),
                     np.expand_dims(self.data_wizard.validation_target_ds[bot, validation_experiment], -1))
                )

                self.model.fit(train_ds,
                               validation_data=validation_ds)

            for experiment in range(self.data_wizard.test_ds.shape[1]):
                print('bot: ' + str(bot) + ' Testing')
                test_ds = tf.data.Dataset.from_tensor_slices(
                    (np.expand_dims(self.data_wizard.test_ds[bot, experiment], 1),
                     np.expand_dims(self.data_wizard.test_target_ds[bot, experiment], -1))
                )
                loss, acc = self.model.evaluate(test_ds)  # returns loss and metrics
                print("loss: %.2f" % loss + "acc: %.2f" % acc)
