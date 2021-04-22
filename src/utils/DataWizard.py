import numpy as np
from abc import abstractmethod
from src.Classes.Swarm import Swarm


class DataWizard:
    def __init__(self, timesteps: int,
                 time_window: int,
                 label_size: int,
                 experiments: list[Swarm],
                 splitting: list[int] = None,
                 preprocessing_type: str = 'raw',
                 data_format: str = 'numpy'):
        if splitting is None:
            splitting = [0.7, 0.2, 0.1]
        self.timesteps: int = timesteps - 1
        self.time_window: int = time_window
        self.label_size: int = label_size
        self.splitting: list[float] = splitting
        self.preprocessing_type: str = preprocessing_type
        self.experiments: list[Swarm] = experiments

        training_experiments = experiments[:int(self.splitting[0] * len(experiments))]
        validation_experiments = experiments[int(self.splitting[0] * len(experiments)):
                                             int((self.splitting[0] + self.splitting[1]) * len(experiments))]
        test_experiments = experiments[int((self.splitting[0] + self.splitting[1]) * len(experiments)):]
        if data_format == 'numpy':
            self.train_ds = self.create_train_numpy_array(training_experiments)
            self.train_target_ds = self.create_target_train_numpy_array(training_experiments)

            self.validation_ds = self.create_val_numpy_array(validation_experiments)
            self.validation_target_ds = self.create_target_val_numpy_array(validation_experiments)

            self.test_ds = self.create_test_numpy_array(test_experiments)
            self.test_target_ds = self.create_test_target_numpy_array(test_experiments)
        else:
            self.train_ds = self.create_train_dataset(training_experiments)
            self.train_target_ds = self.create_target_train_dataset(training_experiments)

            self.validation_ds = self.create_val_dataset(validation_experiments)
            self.validation_target_ds = self.create_target_val_dataset(validation_experiments)

            self.test_ds = self.create_test_dataset(test_experiments)
            self.test_target_ds = self.create_test_target_dataset(test_experiments)

    @staticmethod
    def shortest_experiment_timesteps(experiment_list: list[Swarm]) -> int:
        return min(
            experiment.list_of_footbots[0].number_of_timesteps for experiment in experiment_list
        )

    @staticmethod
    def retrieve_bot_features(bot) -> list[np.ndarray]:
        vector = [bot.single_robot_positions[:, 0],
                  bot.single_robot_positions[:, 1],
                  bot.traversed_distance_time_series,
                  bot.direction_time_series[:, 0],
                  bot.direction_time_series[:, 1],
                  bot.cumulative_traversed_distance,
                  bot.neighbors_time_series,
                  bot.swarm_cohesion_time_series,
                  bot.distance_from_centroid_time_series]
        return vector

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
