from src.classes.Swarm import Swarm
from src.Utils.data_utils.data_wizards.DataWizard import DataWizard


class SwarmDataWizard(DataWizard):
    """
    Class to create datasets_classes of features including global swarm features
    """

    def __init__(self,
                 timesteps: int,
                 time_window: int,
                 feature_set_number: int,
                 experiments: list[Swarm]):

        super().__init__(timesteps=timesteps,
                         time_window=time_window,
                         feature_set_number=feature_set_number,
                         experiments=experiments)
