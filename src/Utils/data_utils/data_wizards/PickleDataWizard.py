import pickle
import random
from os.path import join
from os import listdir, sep
from src.Utils.Parser import Parser
from src.Utils.data_utils.data_wizards.BotDataWizard import BotDataWizard
from src.Utils.Plotter import Plotter
from src.classes.Swarm import Swarm


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
                                               feature_set_name: str,
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

            wizard = BotDataWizard(time_window=self.time_window,
                                   experiments=experiment_list,
                                   down_sampling_steps=self.down_sampling_steps,
                                   feature_set_name=feature_set_name,
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
                                                   feature_set_name: str,
                                                   experiments_downsampling: int,
                                                   delete_useless_bots: bool,
                                                   useless_bot_deletion_factor: int,
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
            print('Loaded swarm: {}'.format(cached_swarm_file.split(sep)[-1]))
            if delete_useless_bots:
                cached_swarm = PickleDataWizard.delete_useless_bots(swarm=cached_swarm,
                                                                    useless_bot_deletion_factor=useless_bot_deletion_factor)
            experiment_list.append(cached_swarm)

        wizard = BotDataWizard(time_window=self.time_window,
                               experiments=experiment_list,
                               down_sampling_steps=self.down_sampling_steps,
                               feature_set_name=feature_set_name,
                               perform_data_balancing=perform_data_balancing)

        datasets = wizard.dataset
        print('Computed Dataset')

        dataset_path = Parser.return_cached_dataset_directory_path(experiment_name=task_name)
        if perform_data_balancing:
            filename = join(dataset_path,
                            '{}_{}exp_features_{}_{}downsampled_balanced.pkl'.format(task_name,
                                                                                     len(cached_swarm_list),
                                                                                     feature_set_name,
                                                                                     self.down_sampling_steps))
        else:
            filename = join(dataset_path,
                            '{}_{}exp_features_{}_{}downsampled_UNbalanced.pkl'.format(task_name,
                                                                                       len(cached_swarm_list),
                                                                                       feature_set_name,
                                                                                       self.down_sampling_steps))
        with open(filename, 'wb') as output_file:
            pickle.dump(datasets, output_file)
        print('saved ' + filename.split(sep)[-1])

    @staticmethod
    def delete_useless_bots(swarm: Swarm,
                            useless_bot_deletion_factor: int):
        faulty_bots, nominal_bots = Plotter.divide_flocks(footbots_list=swarm.list_of_footbots)
        if not faulty_bots:
            random.seed(Parser.read_seed())
            random.shuffle(swarm.list_of_footbots)
            swarm.list_of_footbots = swarm.list_of_footbots[::useless_bot_deletion_factor]
        else:
            faulty_nominal_ratio = int(len(nominal_bots)/len(faulty_bots))
            swarm.list_of_footbots = []
            swarm.list_of_footbots.extend(nominal_bots[::faulty_nominal_ratio])
            swarm.list_of_footbots.extend(faulty_bots)
        return swarm
