import random
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import List, Tuple

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
                 time_window: int,
                 experiments: List[Swarm],
                 feature_set_name: str,
                 perform_data_balancing: bool,
                 down_sampling_steps: int = 1):
        super().__init__(time_window=time_window,
                         experiments=experiments,
                         feature_set_name=feature_set_name,
                         down_sampling_steps=down_sampling_steps)

        self.dataset: GeneralDataset = self.create_balanced_bot_train_test_set(
            experiments=experiments,
            feature_set_name=self.feature_set_name,
            perform_data_balancing=perform_data_balancing
        )
        print('BotDataWizard finished creating balanced datasets_classes')
        # feature are normalized in order to have zero mean and unit variance
        self.normalize_dataset()
        print('BotDataWizard transposed and normalized datasets')

    def create_balanced_bot_train_test_set(self,
                                           experiments: List[Swarm],
                                           feature_set_name: str,
                                           perform_data_balancing: bool,
                                           randomization: bool = False) -> GeneralDataset:

        random.seed(Parser.read_seed())

        datasets = []

        max_bot_number = DataWizard.maximum_bot_number_in_experiment(experiment_list=experiments)

        validation = bool(Parser.read_validation_choice())
        if not validation:
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
                                                                             feature_set_name=feature_set_name,
                                                                             down_sampling_steps=self.
                                                                             down_sampling_steps)
                    datasets.append(bot_dataset)

            feature_names = Parser.read_features_names(feature_set_name=feature_set_name)
            if perform_data_balancing:
                dataset = self.balance_train_test_dataset(datasets,
                                                          feature_names=feature_names)
            else:
                dataset = GeneralDataset(bot_identifier=0,
                                         feature_names=feature_names,
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
                        feature_set_name=feature_set_name,
                        down_sampling_steps=self.down_sampling_steps
                    )
                    datasets.append(bot_dataset)

            feature_names = Parser.read_features_names(feature_set_name=feature_set_name)
            if perform_data_balancing:
                dataset = self.balance_train_test_dataset(datasets,
                                                          feature_names=feature_names)
            else:
                dataset = TrValTeDataset(bot_identifier=0,
                                         feature_names=feature_names,
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
    def normalize_single_feature(dataset: np.ndarray,
                                 scaler: TransformerMixin):
        standardized_dataset = []
        for feature_index in range(dataset.shape[-1]):
            # the slicing at the end of fit_transform is needed to delete the last dimension added by reshape(-1, 1)
            standardized_dataset.append(
                scaler.fit_transform(dataset[..., feature_index].reshape(-1, 1))[..., 0]
            )
        # transposition is necessary because the stacking done in the for cycle features and timesteps gets reversed
        array_after = np.asarray(standardized_dataset).transpose()
        return array_after

    @staticmethod
    def balance_train_test_dataset(datasets_list: List[GeneralDataset],
                                   feature_names: List[str]) -> GeneralDataset:
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

        concatenated_train_datasets, concatenated_target_train_datasets = BotDataWizard.stack_datasets_and_balance(
            max_label=max_label,
            min_label=min_label,
            labels_ratio=labels_ratio,
            concatenated_datasets=concatenated_train_datasets,
            concatenated_target_datasets=concatenated_target_train_datasets
        )

        min_label, max_label, labels_ratio = BotDataWizard.compute_labels_values(concatenated_target_test_datasets)

        concatenated_test_datasets, concatenated_target_test_datasets = BotDataWizard.stack_datasets_and_balance(
            max_label=max_label,
            min_label=min_label,
            labels_ratio=labels_ratio,
            concatenated_datasets=concatenated_test_datasets,
            concatenated_target_datasets=concatenated_target_test_datasets
        )
        return GeneralDataset(bot_identifier=0,
                              feature_names=feature_names,
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

    @staticmethod
    def stack_datasets_and_balance(max_label: bool,
                                   min_label: bool,
                                   labels_ratio: int,
                                   concatenated_datasets: np.ndarray,
                                   concatenated_target_datasets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # check that features and targets match in size
        assert concatenated_datasets.shape[0] == concatenated_target_datasets.shape[0]
        # zip together two numpy arrays
        stacked_train_dataset = np.column_stack((concatenated_datasets, concatenated_target_datasets))
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
        return_concatenated_datasets = stacked_train_dataset[..., :-1]
        # retrieves the target dataset from the stacked array
        return_concatenated_target_datasets = stacked_train_dataset[..., -1]

        return return_concatenated_datasets, return_concatenated_target_datasets
