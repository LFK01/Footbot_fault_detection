import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from src.Classes.Swarm import Swarm
from src.utils.DataWizard import DataWizard


class BotDataWizard(DataWizard):

    def __init__(self, timesteps: int, time_window: int, label_size: int, experiments: list[Swarm],
                 splitting: list[int] = None, preprocessing_type: str = 'raw', data_format: str = 'numpy'):
        super().__init__(timesteps, time_window, label_size, experiments, splitting, preprocessing_type, data_format)

    def create_numpy_array(self, experiments: list[Swarm]):
        dataset_vector = []
        for bot in range(len(experiments[0].list_of_footbots)):
            bot_vector = []
            for exp in experiments:
                exp_bot = exp.list_of_footbots[bot]

                exp_vector = DataWizard.retrieve_bot_features(exp_bot)

                exp_vector = np.asarray(exp_vector)
                bot_vector.append(exp_vector[..., :self.timesteps])

            bot_vector = np.asarray(bot_vector)

            if self.preprocessing_type == 'norm':
                scaler = MinMaxScaler(feature_range=(0, 1))
                bot_vector = scaler.fit_transform(bot_vector)
            if self.preprocessing_type == 'norm':
                scaler = StandardScaler()
                bot_vector = scaler.fit_transform(bot_vector)

            windowed_vector = []
            for exp in bot_vector:
                sliced_array = []
                for i in range(exp.shape[-1] - self.time_window):
                    sliced_array.append(exp[..., i:i+self.time_window])
                windowed_vector.append(sliced_array)

            dataset_vector.append(np.asarray(windowed_vector))

        return dataset_vector

    def prepare_target(self, experiments: list[Swarm]):
        target_vector = []
        for bot in range(len(experiments[0].list_of_footbots)):
            bot_vector = []
            for exp in experiments:
                exp_bot = exp.list_of_footbots[bot]
                bot_vector.append(
                    exp_bot.fault_time_series[..., self.time_window:self.timesteps]
                )
            target_vector.append(bot_vector)

        return target_vector

    def create_train_numpy_array(self, experiments: list[Swarm]) -> np.ndarray:
        return np.asarray(self.create_numpy_array(experiments=experiments))

    def create_target_train_numpy_array(self, experiments) -> np.ndarray:
        return np.asarray(self.prepare_target(experiments=experiments))

    def create_val_numpy_array(self, experiments) -> np.ndarray:
        return np.asarray(self.create_numpy_array(experiments=experiments))

    def create_target_val_numpy_array(self, experiments) -> np.ndarray:
        return np.asarray(self.prepare_target(experiments=experiments))

    def create_test_numpy_array(self, experiments) -> np.ndarray:
        return np.asarray(self.create_numpy_array(experiments=experiments))

    def create_test_target_numpy_array(self, experiments) -> np.ndarray:
        return np.asarray(self.prepare_target(experiments=experiments))

    def create_target_train_dataset(self, experiments):
        pass

    def create_val_dataset(self, experiments):
        pass

    def create_target_val_dataset(self, experiments):
        pass

    def create_test_dataset(self, experiments):
        pass

    def create_test_target_dataset(self, experiments):
        pass

    def create_train_dataset(self, experiments):
        pass
