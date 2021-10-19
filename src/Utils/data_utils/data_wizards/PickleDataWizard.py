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

    def save_bot_train_test_dataset_all_swarms(self):
        experiment_list = []
        root = Parser.get_project_root()
        cached_swarms_path = join(root, 'cached_files', 'cached_swarms')
        for folder in listdir(cached_swarms_path):
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
                                   experiments=experiment_list,
                                   down_sampling_steps=self.down_sampling_steps)

            datasets = wizard.datasets
            print('Computed Dataset')

            filename = join(root, 'cached_files', 'cached_datasets',
                            '{}_{}exp_ALL_features.pkl'.format(folder, len(listdir(swarm_path))))
            with open(filename, 'wb') as output_file:
                pickle.dump(datasets, output_file)
            print('saved ' + filename)

    def save_bot_train_test_dataset_specific_swarm(self, task_name: str):
        experiment_list = []
        root = Parser.get_project_root()
        swarm_path = Parser.return_cached_swarm_directory_path(experiment_name=task_name)
        for cached_swarm_file in Parser.read_cached_swarms_in_directory(experiment_name=task_name):
            with open(join(swarm_path, cached_swarm_file), 'rb') as f:
                cached_swarm = pickle.load(file=f)
            print('Loaded swarm: {}'.format(cached_swarm_file))
            experiment_list.append(cached_swarm)
        timesteps = SwarmDataWizard.shortest_experiment_timesteps(experiment_list=experiment_list)
        wizard = BotDataWizard(timesteps=timesteps,
                               time_window=self.time_window,
                               experiments=experiment_list,
                               down_sampling_steps=self.down_sampling_steps)

        datasets = wizard.datasets
        print('Computed Dataset')

        filename = join(root, 'cached_files', 'cached_datasets',
                        '{}_{}exp_ALL_features_{}downsampled.pkl'.format(task_name,
                                                              len(listdir(swarm_path)),
                                                              self.down_sampling_steps))
        with open(filename, 'wb') as output_file:
            pickle.dump(datasets, output_file)
        print('saved ' + filename)


if __name__ == '__main__':
    down_sampling = Parser.read_down_sampling_size()
    time_window_size = Parser.read_time_window()

    task_name = ''
    pickle_wizard = PickleDataWizard(time_window=time_window_size,
                                     down_sampling_steps=down_sampling)
    pickle_wizard.save_bot_train_test_dataset_specific_swarm(task_name=task_name)
