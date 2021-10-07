import random
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.Utils.data_utils.datasets.GeneralDataset import GeneralDataset
from src.Utils.Parser import Parser
from src.classes.Swarm import Swarm
from src.Utils.data_utils.data_wizards.DataWizard import DataWizard


class BotDataWizard(DataWizard):
    """
    Class to build bots datasets
    """

    def __init__(self,
                 timesteps: int,
                 time_window: int,
                 label_size: int,
                 experiments: list[Swarm],
                 down_sampling_steps: int = 1):
        super().__init__(timesteps=timesteps,
                         time_window=time_window,
                         label_size=label_size,
                         experiments=experiments,
                         down_sampling_steps=down_sampling_steps)

        self.datasets: list[GeneralDataset] = self.create_balanced_bot_train_test_set(
            experiments=experiments
        )
        print('Created balanced datasets')
        # feature are transposed in order to have the timesteps in the first axis and the features in the second
        self.transpose_dataset()
        # feature are normalized in order to have zero mean and unit variance
        self.normalize_dataset()

    def create_balanced_bot_train_test_set(self,
                                           experiments: list[Swarm],
                                           randomization: bool = False) -> list[GeneralDataset]:

        random.seed(Parser.read_seed())
        splitting = Parser.read_dataset_splittings()['No validation']

        dataset = []

        for bot in range(len(experiments[0].list_of_footbots)):
            print('Building feature vectors for bot:' + str(bot))
            fault_experiments = [exp for exp in experiments if any(exp.list_of_footbots[bot].fault_time_series)]
            nominal_experiments = [exp for exp in experiments if not any(exp.list_of_footbots[bot].fault_time_series)]

            train_experiments = nominal_experiments[
                                :int(len(nominal_experiments) * splitting[0])
                                ]
            test_experiments = nominal_experiments[
                               int(len(nominal_experiments) * splitting[0]):
                               ]

            if len(fault_experiments) > 0:
                train_experiments.extend(fault_experiments[
                                         :int(len(fault_experiments) * splitting[0])
                                         ])
                test_experiments.extend(fault_experiments[
                                        int(len(fault_experiments) * splitting[0]):
                                        ])

            if randomization:
                random.shuffle(train_experiments)
                random.shuffle(test_experiments)

            bot_dataset = BotDataWizard.slice_train_test_experiments(bot=bot,
                                                                     train_experiments=train_experiments,
                                                                     test_experiments=test_experiments,
                                                                     down_sampling_steps=self.down_sampling_steps)
            dataset.append(bot_dataset)

        return dataset

    def normalize_dataset(self):
        if Parser.read_validation_choice():
            if Parser.read_preprocessing_type() == 'STD':
                scaler = StandardScaler()
                for bot_dataset in self.datasets:
                    bot_dataset.train_dataset = scaler.fit_transform(bot_dataset.train_dataset)
                    bot_dataset.validation_dataset = scaler.fit_transform(bot_dataset.validation_dataset)
                    bot_dataset.test_dataset = scaler.fit_transform(bot_dataset.test_dataset)
            else:
                scaler = MinMaxScaler(feature_range=(0, 1))
                for bot_dataset in self.datasets:
                    bot_dataset.train_dataset = scaler.fit_transform(bot_dataset.train_dataset)
                    bot_dataset.validation_dataset = scaler.fit_transform(bot_dataset.validation_dataset)
                    bot_dataset.test_dataset = scaler.fit_transform(bot_dataset.test_dataset)
        else:
            if Parser.read_preprocessing_type() == 'STD':
                scaler = StandardScaler()
                for bot_dataset in self.datasets:
                    bot_dataset.train_dataset = scaler.fit_transform(bot_dataset.train_dataset)
                    bot_dataset.test_dataset = scaler.fit_transform(bot_dataset.test_dataset)
            else:
                scaler = MinMaxScaler(feature_range=(0, 1))
                for bot_dataset in self.datasets:
                    bot_dataset.train_dataset = scaler.fit_transform(bot_dataset.train_dataset)
                    bot_dataset.test_dataset = scaler.fit_transform(bot_dataset.test_dataset)

    def transpose_dataset(self):
        if Parser.read_validation_choice():
            for bot_dataset in self.datasets:
                bot_dataset.train_dataset = np.transpose(bot_dataset.train_dataset)
                bot_dataset.validation_dataset = np.transpose(bot_dataset.validation_dataset)
                bot_dataset.test_dataset = np.transpose(bot_dataset.test_dataset)
        else:
            for bot_dataset in self.datasets:
                bot_dataset.train_dataset = np.transpose(bot_dataset.train_dataset)
                bot_dataset.test_dataset = np.transpose(bot_dataset.test_dataset)
