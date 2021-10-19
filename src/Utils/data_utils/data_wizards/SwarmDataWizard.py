from src.classes.Swarm import Swarm
from src.Utils.data_utils.data_wizards.DataWizard import DataWizard


class SwarmDataWizard(DataWizard):
    """
    Class to create datasets_classes of features including global swarm features
    """

    def __init__(self,
                 timesteps: int,
                 time_window: int,
                 experiments: list[Swarm]):

        super().__init__(timesteps, time_window, experiments)
