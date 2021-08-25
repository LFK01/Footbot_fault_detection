import numpy as np


class BotDataset:

    def __init__(self,
                 train_dataset: np.ndarray,
                 target_train_dataset: np.ndarray,
                 validation_dataset: np.ndarray,
                 target_validation_dataset: np.ndarray,
                 test_dataset: np.ndarray,
                 target_test_dataset: np.ndarray):

        self.train_dataset = train_dataset
        self.target_train_dataset = target_train_dataset
        self.validation_dataset = validation_dataset
        self.target_validation_dataset = target_validation_dataset
        self.test_dataset = test_dataset
        self.target_test_dataset = target_test_dataset
