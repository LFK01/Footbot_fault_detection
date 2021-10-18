import pickle
from os.path import join
from os import listdir
from src.Utils.Parser import Parser
from src.Utils.data_utils.data_wizards.BotDataWizard import BotDataWizard
from src.Utils.data_utils.data_wizards.SwarmDataWizard import SwarmDataWizard


class PickleDataWizard:
    """
    Class to build bots datasets_classes exploiting the swarm saved with pickle
    """

    def __init__(self,
                 time_window: int,
                 down_sampling_steps: int = 1):
        self.time_window = time_window
        self.down_sampling_steps = down_sampling_steps
        self.create_balanced_bot_train_test_set()

    def create_balanced_bot_train_test_set(self):
        experiment_list = []
        root = Parser.get_project_root()
        cached_swarms_path = join(root, 'cached_files', 'cached_swarms')
        for folder in listdir(cached_swarms_path)[3:]:
            print('Working in folder: {}'.format(folder))
            swarm_path = join(cached_swarms_path, folder)
            for cached_swarm_file in listdir(swarm_path):
                with open(join(swarm_path, cached_swarm_file), 'rb') as f:
                    cached_swarm = pickle.load(file=f)
                print('Loaded swarm: {}'.format(cached_swarm_file))
                experiment_list.append(cached_swarm)
            timesteps = SwarmDataWizard.shortest_experiment_timesteps(experiment_list=experiment_list)
            wizard = BotDataWizard(timesteps=timesteps,
                                   time_window=self.time_window,
                                   experiments=experiment_list)

            datasets = wizard.datasets
            print('Computed Dataset')

            filename = join(root, 'cached_files', 'cached_datasets',
                            '{}_{}exp_ALL_features.pkl'.format(folder, len(listdir(swarm_path))))
            with open(filename, 'wb') as output_file:
                pickle.dump(datasets, output_file)
            print('saved ' + filename)
