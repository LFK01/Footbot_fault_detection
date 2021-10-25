import pickle
import random
from os.path import join
from os import listdir
from src.Utils.Parser import Parser
from src.Utils.data_utils.data_wizards.BotDataWizard import BotDataWizard
from src.Utils.data_utils.data_wizards.DataWizard import DataWizard


class PickleDataWizard:
    """
    Class to build bots datasets_classes exploiting the swarm saved with pickle
    """

    def __init__(self,
                 time_window: int,
                 down_sampling_steps: int = 1):
        self.time_window = time_window
        self.down_sampling_steps = down_sampling_steps

    def save_bot_train_test_dataset_all_swarms(self,
                                               feature_set_number: int,
                                               perform_data_balancing: bool):
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
            timesteps = DataWizard.shortest_experiment_timesteps(experiment_list=experiment_list)
            wizard = BotDataWizard(timesteps=timesteps,
                                   time_window=self.time_window,
                                   experiments=experiment_list,
                                   down_sampling_steps=self.down_sampling_steps,
                                   feature_set_number=feature_set_number,
                                   perform_data_balancing=perform_data_balancing)

            datasets = wizard.dataset
            print('Computed Dataset')

            filename = join(root, 'cached_files', 'cached_datasets',
                            '{}_{}exp_ALL_features.pkl'.format(folder, len(listdir(swarm_path))))
            with open(filename, 'wb') as output_file:
                pickle.dump(datasets, output_file)
            print('saved ' + filename)

    def save_bot_train_test_dataset_specific_swarm(self,
                                                   task_name: str,
                                                   feature_set_number: int,
                                                   experiments_downsampling: int,
                                                   perform_data_balancing: bool):
        experiment_list = []
        random.seed(Parser.read_seed())

        swarm_path = Parser.return_cached_swarm_directory_path(experiment_name=task_name)
        cached_swarm_list = Parser.read_cached_swarms_in_directory(experiment_name=task_name)

        random.shuffle(cached_swarm_list)
        cached_swarm_list = cached_swarm_list[::experiments_downsampling]
        for cached_swarm_file in cached_swarm_list:
            with open(join(swarm_path, cached_swarm_file), 'rb') as f:
                cached_swarm = pickle.load(file=f)
            print('Loaded swarm: {}'.format(cached_swarm_file))
            experiment_list.append(cached_swarm)
        timesteps = DataWizard.shortest_experiment_timesteps(experiment_list=experiment_list)
        wizard = BotDataWizard(timesteps=timesteps,
                               time_window=self.time_window,
                               experiments=experiment_list,
                               down_sampling_steps=self.down_sampling_steps,
                               feature_set_number=feature_set_number,
                               perform_data_balancing=perform_data_balancing)

        datasets = wizard.dataset
        print('Computed Dataset')

        dataset_path = Parser.return_cached_dataset_directory_path(experiment_name=task_name)
        filename = join(dataset_path,
                        '{}_{}exp_features_set{}_{}downsampled.pkl'.format(task_name,
                                                                           len(cached_swarm_list),
                                                                           feature_set_number,
                                                                           self.down_sampling_steps))
        with open(filename, 'wb') as output_file:
            pickle.dump(datasets, output_file)
        print('saved ' + filename)
