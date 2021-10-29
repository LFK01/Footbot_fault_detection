import numpy as np
from typing import List

from src.Utils.data_utils.datasets_classes.GeneralDataset import GeneralDataset


class TrValTeDataset(GeneralDataset):
    """
    Class to store and organize datasets_classes with test, validation and test splits
    """
    def __init__(self,
                 bot_identifier: int,
                 feature_names: List[str],
                 train_dataset: np.ndarray,
                 target_train_dataset: np.ndarray,
                 test_dataset: np.ndarray,
                 target_test_dataset: np.ndarray,
                 validation_dataset: np.ndarray,
                 target_validation_dataset: np.ndarray):
        super().__init__(bot_identifier=bot_identifier,
                         feature_names=feature_names,
                         train_dataset=train_dataset,
                         target_train_dataset=target_train_dataset,
                         test_dataset=test_dataset,
                         target_test_dataset=target_test_dataset)
        self.validation_dataset = validation_dataset
        self.target_validation_dataset = target_validation_dataset
