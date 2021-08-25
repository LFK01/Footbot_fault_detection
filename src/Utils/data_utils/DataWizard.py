from abc import abstractmethod
import numpy as np
import random
from src.utils.Parser import Parser
from src.Classes.Swarm import Swarm
from src.Classes.FootBot import FootBot
from src.utils.data_utils.BotDataset import BotDataset


class DataWizard:
    def __init__(self, timesteps: int,
                 time_window: int,
                 label_size: int,
                 experiments: list[Swarm],
                 down_sampling_steps: int = 1,
                 preprocessing_type: str = 'raw'):
        self.timesteps: int = timesteps - 1
        self.time_window: int = time_window
        self.label_size: int = label_size
        self.experiments: list[Swarm] = experiments
        self.down_sampling_steps = down_sampling_steps
        self.preprocessing_type = preprocessing_type

    @staticmethod
    def shortest_experiment_timesteps(experiment_list: list[Swarm]) -> int:
        return min(
            experiment.list_of_footbots[0].number_of_timesteps for experiment in experiment_list
        )

    @staticmethod
    def flatten_experiments(dataset: np.ndarray):
        return np.concatenate([exp for exp in dataset])

    @staticmethod
    def create_balanced_bot_train_val_test_set(experiments: list[Swarm],
                                               splitting=None,
                                               down_sampling_steps=1) -> list[BotDataset]:
        random.seed(Parser.read_seed())
        if splitting is None:
            splitting = [0.7, 0.2, 0.1]

        dataset = []

        for bot in range(len(experiments[0].list_of_footbots)):
            fault_experiments = [exp for exp in experiments if any(exp.list_of_footbots[bot].fault_time_series)]
            nominal_experiments = [exp for exp in experiments if exp not in fault_experiments]

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
                                              int(len(fault_experiments) * splitting[0]):
                                              int(len(fault_experiments) * (splitting[0] + splitting[1]))
                                              ])
                test_experiments.extend(fault_experiments[
                                        int(len(fault_experiments) * (splitting[0] + splitting[1])):
                                        ])

            random.shuffle(train_experiments)
            random.shuffle(validation_experiments)
            random.shuffle(test_experiments)

            bot_dataset = DataWizard.slice_experiments(bot=bot,
                                                       train_experiments=train_experiments,
                                                       validation_experiments=validation_experiments,
                                                       test_experiments=test_experiments,
                                                       down_sampling_steps=down_sampling_steps)
            dataset.append(bot_dataset)

        return dataset

    @staticmethod
    def slice_experiments(bot: int,
                          train_experiments,
                          validation_experiments,
                          test_experiments,
                          down_sampling_steps: int = 1) -> BotDataset:

        bot_train_dataset, bot_train_target_dataset = DataWizard.retrieve_features_and_target(
            bot=bot,
            experiments=train_experiments,
            down_sampling_steps=down_sampling_steps
        )
        bot_val_dataset, bot_val_target_dataset = DataWizard.retrieve_features_and_target(
            bot=bot,
            experiments=validation_experiments,
            down_sampling_steps=down_sampling_steps
        )
        bot_test_dataset, bot_test_target_dataset = DataWizard.retrieve_features_and_target(
            bot=bot,
            experiments=test_experiments,
            down_sampling_steps=down_sampling_steps
        )

        bot_dataset = BotDataset(
            train_dataset=np.concatenate(bot_train_dataset, axis=-1),
            target_train_dataset=np.concatenate(bot_train_target_dataset, axis=-1),
            validation_dataset=np.concatenate(bot_val_dataset, axis=-1),
            target_validation_dataset=np.concatenate(bot_val_target_dataset, axis=-1),
            test_dataset=np.concatenate(bot_test_dataset, axis=-1),
            target_test_dataset=np.concatenate(bot_test_target_dataset, axis=-1)
        )
        return bot_dataset

    @staticmethod
    def retrieve_bot_features(bot: FootBot, down_sampling_steps: int) -> list[np.ndarray]:
        vector = [
            bot.single_robot_positions[::down_sampling_steps, 0],
            bot.single_robot_positions[::down_sampling_steps, 1],
            bot.traversed_distance_time_series[::down_sampling_steps],
            bot.direction_time_series[::down_sampling_steps, 0],
            bot.direction_time_series[::down_sampling_steps, 1],
            bot.cumulative_traversed_distance[::down_sampling_steps],
            bot.neighbors_time_series[::down_sampling_steps],
            bot.swarm_cohesion_time_series[::down_sampling_steps],
            bot.distance_from_centroid_time_series[::down_sampling_steps]
        ]
        # since some arrays may end up having different lengths, we short them to the maximum common length
        # which is the minimum among the length of all the features
        cut_max_common_length = min(feature.shape[0] for feature in vector)
        vector = [feature[:cut_max_common_length] for feature in vector]
        return vector

    @staticmethod
    def retrieve_features_and_target(bot: int, experiments: list[Swarm], down_sampling_steps: int = 1):
        bot_dataset = []
        bot_target_dataset = []

        for exp in experiments:
            retrieved_features = np.asarray(DataWizard.retrieve_bot_features(bot=exp.list_of_footbots[bot],
                                                                             down_sampling_steps=down_sampling_steps))

            bot_dataset.append(retrieved_features)
            bot_target_dataset.append(exp.list_of_footbots[bot].fault_time_series[::down_sampling_steps])

        return bot_dataset, bot_target_dataset

    @staticmethod
    def standard_normalization(dataset: list[list[np.ndarray]]):
        pass

    @abstractmethod
    def create_train_numpy_array(self, experiments: list[Swarm]) -> np.ndarray:
        """abstract method"""

    @abstractmethod
    def create_target_train_numpy_array(self, experiments) -> np.ndarray:
        """abstract method"""

    @abstractmethod
    def create_val_numpy_array(self, experiments) -> np.ndarray:
        """abstract method"""

    @abstractmethod
    def create_target_val_numpy_array(self, experiments) -> np.ndarray:
        """abstract method"""

    @abstractmethod
    def create_test_numpy_array(self, experiments) -> np.ndarray:
        """abstract method"""

    @abstractmethod
    def create_test_target_numpy_array(self, experiments) -> np.ndarray:
        """abstract method"""

    @abstractmethod
    def create_train_dataset(self, experiments):
        """abstract method"""

    @abstractmethod
    def create_target_train_dataset(self, experiments):
        """abstract method"""

    @abstractmethod
    def create_val_dataset(self, experiments):
        """abstract method"""

    @abstractmethod
    def create_target_val_dataset(self, experiments):
        """abstract method"""

    @abstractmethod
    def create_test_dataset(self, experiments):
        """abstract method"""

    @abstractmethod
    def create_test_target_dataset(self, experiments):
        """abstract method"""
