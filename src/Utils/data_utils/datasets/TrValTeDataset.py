import numpy as np

from src.Utils.data_utils.datasets.GeneralDataset import GeneralDataset


class TrValTeDataset(GeneralDataset):
    """
    Class to store and organize datasets with test, validation and test splits
    """
    def __init__(self,
                 bot_identifier: int,
                 train_dataset: np.ndarray,
                 target_train_dataset: np.ndarray,
                 test_dataset: np.ndarray,
                 target_test_dataset: np.ndarray,
                 validation_dataset: np.ndarray,
                 target_validation_dataset: np.ndarray):
        super().__init__(bot_identifier=bot_identifier,
                         train_dataset=train_dataset,
                         target_train_dataset=target_train_dataset,
                         test_dataset=test_dataset,
                         target_test_dataset=target_test_dataset)
        self.validation_dataset = validation_dataset
        self.target_validation_dataset = target_validation_dataset
