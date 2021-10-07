import numpy as np


class GeneralDataset:
    """
    Class to store a generic dataset split in train and test
    """

    def __init__(self,
                 bot_identifier: int,
                 train_dataset: np.ndarray,
                 target_train_dataset: np.ndarray,
                 test_dataset: np.ndarray,
                 target_test_dataset: np.ndarray):
        self.bot_identifier: int = bot_identifier
        self.train_dataset: np.ndarray = train_dataset
        self.target_train_dataset: np.ndarray = target_train_dataset
        self.test_dataset: np.ndarray = test_dataset
        self.target_test_dataset: np.ndarray = target_test_dataset
