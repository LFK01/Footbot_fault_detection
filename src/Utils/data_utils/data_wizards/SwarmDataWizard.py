from src.classes.Swarm import Swarm
from src.Utils.data_utils.data_wizards.DataWizard import DataWizard


class SwarmDataWizard(DataWizard):
    """
    Class to create datasets of features including global swarm features
    """

    def __init__(self,
                 timesteps: int,
                 time_window: int,
                 label_size: int,
                 experiments: list[Swarm]):

        super().__init__(timesteps, time_window, label_size, experiments)
