import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from src.Classes.Swarm import Swarm
from src.utils.data_utils.DataWizard import DataWizard


class SwarmDataWizard(DataWizard):
    def __init__(self, timesteps: int, time_window: int, label_size: int, experiments: list[Swarm]):

        super().__init__(timesteps, time_window, label_size, experiments)

    @staticmethod
    def build_dataset_vector(timesteps, preprocessing_type, experiments) -> list[list[np.ndarray]]:
        datasets_vector = []
        for swarm in experiments:
            data_vector = []
            # append all robots features
            for bot in swarm.list_of_footbots:
                data_vector.append(SwarmDataWizard.retrieve_bot_features(bot))

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
        datasets_vector = SwarmDataWizard.build_dataset_vector(timesteps=timesteps,
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
        datasets_vector = SwarmDataWizard.build_dataset_vector(timesteps=timesteps,
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
