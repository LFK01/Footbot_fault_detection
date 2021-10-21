import random
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.Utils.data_utils.datasets_classes.GeneralDataset import GeneralDataset
from src.Utils.Parser import Parser
from src.classes.Swarm import Swarm
from src.Utils.data_utils.data_wizards.DataWizard import DataWizard


class BotDataWizard(DataWizard):
    """
    Class to build bots datasets_classes
    """

    def __init__(self,
                 timesteps: int,
                 time_window: int,
                 experiments: list[Swarm],
                 feature_set_number: int,
                 down_sampling_steps: int = 1):
        super().__init__(timesteps=timesteps,
                         time_window=time_window,
                         experiments=experiments,
                         feature_set_number=feature_set_number,
                         down_sampling_steps=down_sampling_steps)

        self.datasets: list[GeneralDataset] = self.create_balanced_bot_train_test_set(
            experiments=experiments,
            feature_set_number=self.feature_set_number
        )
        print('BotDataWizard finished creating balanced datasets_classes')
        # feature are transposed in order to have the timesteps in the first axis and the features in the second
        self.transpose_dataset()
        # feature are normalized in order to have zero mean and unit variance
        self.normalize_dataset()
        print('BotDataWizard transposed and normalized datasets')

    def create_balanced_bot_train_test_set(self,
                                           experiments: list[Swarm],
                                           feature_set_number: int,
                                           randomization: bool = False) -> list[GeneralDataset]:

        random.seed(Parser.read_seed())
        splitting = Parser.read_dataset_splittings()['No validation']

        dataset = []

        max_bot_number = DataWizard.maximum_bot_number_in_experiment(experiment_list=experiments)
        for bot_index in range(max_bot_number):
            print('Building feature vectors for bot:' + str(bot_index))
            feasible_experiments = [exp for exp in experiments
                                    if len(exp.list_of_footbots) > bot_index]

            if len(feasible_experiments) > 2:
                fault_experiments = [exp for exp in feasible_experiments
                                     if any(exp.list_of_footbots[bot_index].fault_time_series)]
                nominal_experiments = [exp for exp in feasible_experiments
                                       if not any(exp.list_of_footbots[bot_index].fault_time_series)]

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

                bot_dataset = BotDataWizard.slice_train_test_experiments(bot=bot_index,
                                                                         train_experiments=train_experiments,
                                                                         test_experiments=test_experiments,
                                                                         feature_set_number=feature_set_number,
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
