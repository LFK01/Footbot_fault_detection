import random
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.Utils.data_utils.datasets_classes.GeneralDataset import GeneralDataset
from src.Utils.data_utils.datasets_classes.TrValTeDataset import TrValTeDataset
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
                 perform_data_balancing: bool,
                 down_sampling_steps: int = 1):
        super().__init__(timesteps=timesteps,
                         time_window=time_window,
                         experiments=experiments,
                         feature_set_number=feature_set_number,
                         down_sampling_steps=down_sampling_steps)

        self.dataset: GeneralDataset = self.create_balanced_bot_train_test_set(
            experiments=experiments,
            feature_set_number=self.feature_set_number,
            perform_data_balancing=perform_data_balancing
        )
        print('BotDataWizard finished creating balanced datasets_classes')
        # feature are normalized in order to have zero mean and unit variance
        self.normalize_dataset()
        print('BotDataWizard transposed and normalized datasets')

    def create_balanced_bot_train_test_set(self,
                                           experiments: list[Swarm],
                                           feature_set_number: int,
                                           perform_data_balancing: bool,
                                           randomization: bool = False) -> GeneralDataset:

        random.seed(Parser.read_seed())

        datasets = []

        max_bot_number = DataWizard.maximum_bot_number_in_experiment(experiment_list=experiments)

        if Parser.read_validation_choice():
            splitting = Parser.read_dataset_splittings()['No validation']
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
                                                                             down_sampling_steps=self.
                                                                             down_sampling_steps)
                    datasets.append(bot_dataset)

            if perform_data_balancing:
                dataset = self.balance_dataset(datasets)
            else:
                dataset = GeneralDataset(bot_identifier=0,
                                         train_dataset=np.concatenate(
                                             [dataset.train_dataset for dataset in datasets], axis=0),
                                         target_train_dataset=np.concatenate(
                                             [dataset.target_train_dataset for dataset in datasets], axis=0),
                                         test_dataset=np.concatenate(
                                             [dataset.test_dataset for dataset in datasets], axis=0),
                                         target_test_dataset=np.concatenate(
                                             [dataset.target_test_dataset for dataset in datasets], axis=0))

        else:
            splitting = Parser.read_dataset_splittings()['With validation']

            for bot_index in range(max_bot_number):
                print('Building feature vectors for bot:' + str(bot_index))
                feasible_experiments = [exp for exp in experiments
                                        if len(exp.list_of_footbots) > bot_index]
                if len(feasible_experiments) > 3:
                    fault_experiments = [exp for exp in feasible_experiments
                                         if any(exp.list_of_footbots[bot_index].fault_time_series)]
                    nominal_experiments = [exp for exp in feasible_experiments
                                           if not any(exp.list_of_footbots[bot_index].fault_time_series)]

                    if randomization:
                        random.shuffle(nominal_experiments)
                        random.shuffle(fault_experiments)

                    train_experiments = nominal_experiments[
                                        :int(len(nominal_experiments) * splitting[0])
                                        ]
                    validation_experiments = nominal_experiments[
                                             int(len(nominal_experiments) * splitting[0]):
                                             int(len(nominal_experiments) * (splitting[0] + splitting[1]))
                                             ]
                    test_experiments = nominal_experiments[
                                       int(len(nominal_experiments) * (splitting[0] + splitting[1])):
                                       ]

                    if len(fault_experiments) > 0:
                        train_experiments.extend(fault_experiments[
                                                 :int(len(fault_experiments) * splitting[0])
                                                 ])
                        validation_experiments.extend(fault_experiments[
                                                      int(len(fault_experiments) * (splitting[0])):
                                                      int(len(fault_experiments) * (splitting[0] + splitting[1]))
                                                      ])
                        test_experiments.extend(fault_experiments[
                                                int(len(fault_experiments) * (splitting[0] + splitting[1])):
                                                ])

                    bot_dataset = BotDataWizard.slice_train_val_test_experiments(
                        bot=bot_index,
                        train_experiments=train_experiments,
                        validation_experiments=validation_experiments,
                        test_experiments=test_experiments,
                        feature_set_number=feature_set_number,
                        down_sampling_steps=self.down_sampling_steps
                    )
                    datasets.append(bot_dataset)
            if perform_data_balancing:
                dataset = self.balance_dataset(datasets)
            else:
                dataset = TrValTeDataset(bot_identifier=0,
                                         train_dataset=np.concatenate(
                                             [dataset.train_dataset for dataset in datasets], axis=0),
                                         target_train_dataset=np.concatenate(
                                             [dataset.target_train_dataset for dataset in datasets], axis=0),
                                         validation_dataset=np.concatenate(
                                             [dataset.validation_dataset for dataset in datasets], axis=0),
                                         target_validation_dataset=np.concatenate(
                                             [dataset.target_validation_dataset for dataset in datasets], axis=0),
                                         test_dataset=np.concatenate(
                                             [dataset.test_dataset for dataset in datasets], axis=0),
                                         target_test_dataset=np.concatenate(
                                             [dataset.target_test_dataset for dataset in datasets], axis=0))
        return dataset

    def normalize_dataset(self):
        if Parser.read_validation_choice():
            if Parser.read_preprocessing_type() == 'STD':
                scaler = StandardScaler()
                self.dataset.train_dataset = scaler.fit_transform(self.dataset.train_dataset)
                self.dataset.validation_dataset = scaler.fit_transform(self.dataset.validation_dataset)
                self.dataset.test_dataset = scaler.fit_transform(self.dataset.test_dataset)
            else:
                scaler = MinMaxScaler(feature_range=(0, 1))
                self.dataset.train_dataset = scaler.fit_transform(self.dataset.train_dataset)
                self.dataset.validation_dataset = scaler.fit_transform(self.dataset.validation_dataset)
                self.dataset.test_dataset = scaler.fit_transform(self.dataset.test_dataset)
        else:
            if Parser.read_preprocessing_type() == 'STD':
                scaler = StandardScaler()
                self.dataset.train_dataset = scaler.fit_transform(self.dataset.train_dataset)
                self.dataset.test_dataset = scaler.fit_transform(self.dataset.test_dataset)
            else:
                scaler = MinMaxScaler(feature_range=(0, 1))
                self.dataset.train_dataset = scaler.fit_transform(self.dataset.train_dataset)
                self.dataset.test_dataset = scaler.fit_transform(self.dataset.test_dataset)

    @staticmethod
    def balance_dataset(datasets_list: list[GeneralDataset]) -> GeneralDataset:
        # TODO balance train val test
        # concatenate all the train features and targets dataset in order to work on an unique time series
        concatenated_train_datasets = np.concatenate(
            [dataset.train_dataset for dataset in datasets_list], axis=0)
        concatenated_target_train_datasets = np.concatenate(
            [dataset.target_train_dataset for dataset in datasets_list], axis=0)
        concatenated_test_datasets = np.concatenate(
            [dataset.test_dataset for dataset in datasets_list], axis=0)
        concatenated_target_test_datasets = np.concatenate(
            [dataset.target_test_dataset for dataset in datasets_list], axis=0)
        min_label, max_label, labels_ratio = BotDataWizard.compute_labels_values(concatenated_target_train_datasets)

        # check that features and targets match in size
        assert concatenated_train_datasets.shape[0] == concatenated_target_train_datasets.shape[0]
        # zip together two numpy arrays
        stacked_train_dataset = np.column_stack((concatenated_train_datasets, concatenated_target_train_datasets))
        # concatenates the two arrays of each class with the overweight array now being downsampled according to the
        # labels ratio number
        stacked_train_dataset = np.concatenate(
            [[sample for sample in stacked_train_dataset if sample[-1] == max_label][::labels_ratio],
             [sample for sample in stacked_train_dataset if sample[-1] == min_label]],
            axis=0
        )
        # shuffles the arrays in order to have samples with different label scattered along the array
        np.random.seed(Parser.read_seed())
        np.random.shuffle(stacked_train_dataset)
        # retrieves the features dataset from the stacked array
        concatenated_train_datasets = stacked_train_dataset[..., :-1]
        # retrieves the target dataset from the stacked array
        concatenated_target_train_datasets = stacked_train_dataset[..., -1]

        min_label, max_label, labels_ratio = BotDataWizard.compute_labels_values(concatenated_target_test_datasets)

        assert concatenated_test_datasets.shape[0] == concatenated_target_test_datasets.shape[0]
        stacked_test_dataset = np.column_stack((concatenated_test_datasets, concatenated_target_test_datasets))
        stacked_test_dataset = np.concatenate(
            [[sample for sample in stacked_test_dataset if sample[-1] == max_label][::labels_ratio],
             [sample for sample in stacked_test_dataset if sample[-1] == min_label]],
            axis=0
        )
        np.random.seed(Parser.read_seed())
        np.random.shuffle(stacked_test_dataset)

        concatenated_test_datasets = stacked_test_dataset[..., :-1]
        concatenated_target_test_datasets = stacked_test_dataset[..., -1]
        return GeneralDataset(bot_identifier=0,
                              train_dataset=concatenated_train_datasets,
                              target_train_dataset=concatenated_target_train_datasets,
                              test_dataset=concatenated_test_datasets,
                              target_test_dataset=concatenated_target_test_datasets)

    @staticmethod
    def compute_labels_values(concatenated_target_dataset: np.ndarray):
        # returns the unique labels in the target array and the count number of each one
        labels, labels_count = np.unique(concatenated_target_dataset, return_counts=True)
        # zips together label and count number to retrieve min label and max labels
        labels_dict = dict(zip(labels, labels_count))
        # retrieves min label and max label from dictionary
        min_label = [key for key in labels_dict.keys() if labels_dict[key] == min(labels_count)][0]
        max_label = [key for key in labels_dict.keys() if labels_dict[key] == max(labels_count)][0]
        # computes the ratio to downsampled the underweight class
        labels_ratio = int(labels_dict[max_label] / labels_dict[min_label])

        return min_label, max_label, labels_ratio
