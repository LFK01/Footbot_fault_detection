import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from src.Classes.Swarm import Swarm
from src.utils.data_utils.DataWizard import DataWizard
from src.utils.data_utils.BotDataset import BotDataset


class BotDataWizard(DataWizard):

    def __init__(self,
                 timesteps: int,
                 time_window: int,
                 label_size: int,
                 experiments: list[Swarm],
                 down_sampling_steps: int = 1,
                 splitting: list[int] = None,
                 preprocessing_type: str = 'raw',
                 data_format: str = 'numpy'):
        super().__init__(timesteps=timesteps,
                         time_window=time_window,
                         label_size=label_size,
                         experiments=experiments,
                         down_sampling_steps=down_sampling_steps,
                         preprocessing_type=preprocessing_type)

        if data_format == 'numpy':
            self.datasets: list[BotDataset] = DataWizard.create_balanced_bot_train_val_test_set(
                experiments=experiments,
                splitting=splitting,
                down_sampling_steps=self.down_sampling_steps
            )
            self.normalize_dataset(self.datasets)
            self.window_and_trim_datasets()
        else:
            self.datasets = DataWizard.create_balanced_bot_train_val_test_set(
                experiments=experiments, splitting=splitting)

    def create_numpy_array(self, experiments: list[Swarm]):
        dataset_vector = []
        for bot in range(len(experiments[0].list_of_footbots)):
            bot_vector = []
            for exp in experiments:
                exp_bot = exp.list_of_footbots[bot]

                exp_vector = DataWizard.retrieve_bot_features(exp_bot)

                exp_vector = np.asarray(exp_vector)[..., :self.timesteps]

                if self.preprocessing_type == 'norm':
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    exp_vector = scaler.fit_transform(exp_vector)
                if self.preprocessing_type == 'std':
                    scaler = StandardScaler()
                    exp_vector = scaler.fit_transform(exp_vector)

                bot_vector.append(exp_vector)

            bot_vector = np.asarray(bot_vector)

            windowed_vector = []
            for exp in bot_vector:
                sliced_array = []
                for i in range(exp.shape[-1] - self.time_window):
                    sliced_array.append(exp[..., i:i + self.time_window])
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

    def window_and_trim_datasets(self):
        false_indexes = list(range(self.time_window))
        for bot_dataset in self.datasets:
            bot_dataset.train_dataset = self.window_numpy_array(bot_dataset.train_dataset)
            mask = np.ones(len(bot_dataset.target_train_dataset), dtype=bool)
            mask[false_indexes] = False
            bot_dataset.target_train_dataset = bot_dataset.target_train_dataset[..., mask]

            bot_dataset.validation_dataset = self.window_numpy_array(bot_dataset.validation_dataset)
            mask = np.ones(len(bot_dataset.target_validation_dataset), dtype=bool)
            mask[false_indexes] = False
            bot_dataset.target_validation_dataset = bot_dataset.target_validation_dataset[..., mask]

            bot_dataset.test_dataset = self.window_numpy_array(bot_dataset.test_dataset)
            mask = np.ones(len(bot_dataset.target_test_dataset), dtype=bool)
            mask[false_indexes] = False
            bot_dataset.target_test_dataset = bot_dataset.target_test_dataset[..., mask]

    def window_numpy_array(self, numpy_array: np.ndarray):
        sliced_array = []
        for i in range(numpy_array.shape[-1] - self.time_window):
            sliced_array.append(numpy_array[..., i:i + self.time_window])
        return np.asarray(sliced_array)

    def normalize_dataset(self, dataset: list[BotDataset]):
        if self.preprocessing_type == 'std':
            scaler = StandardScaler()
            for bot_dataset in dataset:
                bot_dataset.train_dataset = scaler.fit_transform(bot_dataset.train_dataset)
                bot_dataset.validation_dataset = scaler.fit_transform(bot_dataset.validation_dataset)
                bot_dataset.test_dataset = scaler.fit_transform(bot_dataset.test_dataset)
        else:
            scaler = MinMaxScaler(feature_range=(0, 1))
            for bot_dataset in dataset:
                bot_dataset.train_dataset = scaler.fit_transform(bot_dataset.train_dataset)
                bot_dataset.validation_dataset = scaler.fit_transform(bot_dataset.validation_dataset)
                bot_dataset.test_dataset = scaler.fit_transform(bot_dataset.test_dataset)

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
