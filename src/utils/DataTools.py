import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
        self.timesteps = timesteps - 1
        self.time_window = time_window
        self.label_size = label_size
        self.splitting = splitting
        self.preprocessing_type = preprocessing_type
        self.experiments = experiments
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
    def build_dataset_vector(timesteps, preprocessing_type, experiments) -> list[list[np.ndarray]]:
        datasets_vector = []
        for swarm in experiments:
            data_vector = []
            # append all robots features
            for bot in swarm.list_of_footbots:
                data_vector.append(bot.single_robot_positions[:, 0])
                data_vector.append(bot.single_robot_positions[:, 1])
                data_vector.append(bot.traversed_distance_time_series)
                data_vector.append(bot.direction_time_series[:, 0])
                data_vector.append(bot.direction_time_series[:, 1])
                data_vector.append(bot.cumulative_traversed_distance)
                data_vector.append(bot.neighbors_time_series)
                data_vector.append(bot.swarm_cohesion_time_series)
                data_vector.append(bot.distance_from_centroid_time_series)

            data_vector.append(swarm.trajectory[:, 0])
            data_vector.append(swarm.trajectory[:, 1])
            data_vector.append(swarm.traversed_distance_time_series)
            data_vector = np.asarray(data_vector)
            data_vector = data_vector[..., :timesteps]

            datasets_vector.append(data_vector)

        if preprocessing_type == 'norm':
            scaler = MinMaxScaler(feature_range=(-1, 1))
            datasets_vector = [scaler.fit_transform(vector) for vector in datasets_vector]
        if preprocessing_type == 'std':
            scaler = StandardScaler()
            datasets_vector = [scaler.fit_transform(vector) for vector in datasets_vector]
        return datasets_vector

    @staticmethod
    def build_numpy_array(timesteps: int, time_window: int, preprocessing_type: str, experiments: list[Swarm]):
        datasets_vector = DataWizard.build_dataset_vector(timesteps=timesteps,
                                                          preprocessing_type=preprocessing_type,
                                                          experiments=experiments)
        datasets_vector = np.asarray(datasets_vector)
        windowed_vector = []
        for exp in range(datasets_vector.shape[0]):
            sliced_array = []
            for i in range(datasets_vector.shape[-1] - time_window):
                sliced_array.append(datasets_vector[exp, ..., i:i+time_window])
            windowed_vector.append(sliced_array)
        return np.asarray(windowed_vector)

    @staticmethod
    def build_tensor_datasets(timesteps: int, preprocessing_type: str, experiments: list[Swarm]):
        datasets_vector = DataWizard.build_dataset_vector(timesteps=timesteps,
                                                          preprocessing_type=preprocessing_type,
                                                          experiments=experiments)
        ds = tf.data.Dataset.from_tensors(datasets_vector)
        return ds

    def prepare_target(self, experiments):
        targets = []
        for swarm in experiments:
            exp_targets = []
            for bot in swarm.list_of_footbots:
                exp_targets.append(bot.fault_time_series[..., self.time_window:self.timesteps])
            targets.append(np.asarray(exp_targets).transpose())
        return targets

    def prepare_target_numpy(self, experiments):
        return np.asarray(self.prepare_target(experiments))

    def prepare_tensor_dataset_target(self, experiments):
        return tf.data.Dataset.from_tensors(self.prepare_target(experiments))

    def create_train_dataset(self, experiments):
        return self.build_tensor_datasets(timesteps=self.timesteps,
                                          preprocessing_type=self.preprocessing_type,
                                          experiments=experiments).window(size=self.time_window, shift=1)

    def create_target_train_dataset(self, experiments):
        return self.prepare_tensor_dataset_target(experiments)

    def create_val_dataset(self, experiments):
        return self.build_tensor_datasets(timesteps=self.timesteps,
                                          preprocessing_type=self.preprocessing_type,
                                          experiments=experiments).window(self.time_window, shift=1)

    def create_target_val_dataset(self, experiments):
        return self.prepare_tensor_dataset_target(experiments)

    def create_test_dataset(self, experiments):
        return self.build_tensor_datasets(timesteps=self.timesteps,
                                          preprocessing_type=self.preprocessing_type,
                                          experiments=experiments).window(self.time_window, shift=1)

    def create_test_target_dataset(self, experiments):
        return self.prepare_tensor_dataset_target(experiments)

    def create_train_numpy_array(self, experiments):
        return self.build_numpy_array(timesteps=self.timesteps,
                                      time_window=self.time_window,
                                      preprocessing_type=self.preprocessing_type,
                                      experiments=experiments)

    def create_target_train_numpy_array(self, experiments):
        return self.prepare_target_numpy(experiments)

    def create_val_numpy_array(self, experiments):
        return self.build_numpy_array(timesteps=self.timesteps,
                                      time_window=self.time_window,
                                      preprocessing_type=self.preprocessing_type,
                                      experiments=experiments)

    def create_target_val_numpy_array(self, experiments):
        return self.prepare_target_numpy(experiments)

    def create_test_numpy_array(self, experiments):
        return self.build_numpy_array(timesteps=self.timesteps,
                                      time_window=self.time_window,
                                      preprocessing_type=self.preprocessing_type,
                                      experiments=experiments)

    def create_test_target_numpy_array(self, experiments):
        return self.prepare_target_numpy(experiments)
