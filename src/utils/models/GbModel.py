import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from src.utils.data_utils.BotDataset import BotDataset
from src.utils.Parser import Parser


class GbModel:
    def __init__(self, datasets: list[BotDataset]):
        self.datasets: list[BotDataset] = datasets
        self.model: GradientBoostingClassifier = self.build_model()

    @staticmethod
    def build_model() -> GradientBoostingClassifier:
        seed = Parser.read_seed()
        model = GradientBoostingClassifier(n_estimators=100,
                                           learning_rate=0.1,
                                           max_depth=3,
                                           random_state=seed,
                                           verbose=2)
        return model

    def train(self):

        train_dataset, target_dataset = self.prepare_train_dataset()

        self.model.fit(X=train_dataset, y=target_dataset)

        test_dataset, target_test_dataset = self.prepare_test_dataset()

        score = self.model.score(X=test_dataset, y=target_test_dataset)
        print('score: ' + str(score))

    @staticmethod
    def flatten_dataset(array: np.ndarray):
        return np.reshape(
            array, (array.shape[0], array.shape[1] * array.shape[2])
        )

    @staticmethod
    def training_condition(bot_dataset: BotDataset) -> bool:
        return any(bot_dataset.target_train_dataset)

    def prepare_train_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        merged_train_dataset = np.concatenate(
            [self.flatten_dataset(bot_dataset.train_dataset) for bot_dataset in self.datasets]
        )
        merged_target_train_dataset = np.concatenate(
            [bot_dataset.target_train_dataset for bot_dataset in self.datasets]
        )

        merged_validation_dataset = np.concatenate(
            [self.flatten_dataset(bot_dataset.validation_dataset) for bot_dataset in self.datasets]
        )
        merged_target_validation_dataset = np.concatenate(
            [bot_dataset.target_validation_dataset for bot_dataset in self.datasets]
        )

        train_dataset = np.concatenate([merged_train_dataset, merged_validation_dataset], axis=0)
        target_dataset = np.concatenate([merged_target_train_dataset, merged_target_validation_dataset], axis=0)

        return train_dataset, target_dataset

    def prepare_test_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        merged_test_dataset = np.concatenate(
            [self.flatten_dataset(bot_dataset.test_dataset) for bot_dataset in self.datasets]
        )
        merged_target_test_dataset = np.concatenate(
            [bot_dataset.target_test_dataset for bot_dataset in self.datasets]
        )

        return merged_test_dataset, merged_target_test_dataset
