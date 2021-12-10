import pickle
import random
from datetime import datetime
from os import listdir, sep
from os.path import join
from typing import Dict, List
from src.Utils.Parser import Parser
from src.classes.Swarm import Swarm
from src.Utils.data_utils.data_wizards.DataWizard import DataWizard
from src.Utils.data_utils.data_wizards.BotDataWizard import BotDataWizard
from src.Utils.data_utils.data_wizards.PickleDataWizard import PickleDataWizard


def build_swarm_no_foraging_stats(task_name: str,
                                  feature_set_name: str,
                                  feature_set_feature_list: List[str],
                                  do_swarm_saving: bool,
                                  experiments_number_down_sampling: int = 1):
    delta_times = []
    neighborhood_radius = Parser.read_neighborhood_radius()
    time_window_size = Parser.read_time_window()

    file_list = Parser.read_files_in_log_files_directory(experiment_name=task_name)
    random.seed(Parser.read_seed())
    done_files = 1
    random.shuffle(file_list)
    file_list = file_list[::experiments_number_down_sampling]
    for file in file_list:
        start_time = datetime.now()
        print('Doing file {} out of {}: {}'.format(done_files, len(file_list), file.split(sep)[-1]))
        print('Feature set: {}'.format(feature_set_name))
        print('Start time: {}'.format(start_time.strftime('%H:%M %d/%m/%Y')))
        footbots_list = Parser.create_generic_swarm(task_name=task_name,
                                                    filename=file,
                                                    neighborhood_radius=neighborhood_radius,
                                                    time_window_size=time_window_size,
                                                    feature_set_features_list=feature_set_feature_list)

        timesteps = Parser.retrieve_timesteps_series_from_dataframe(
            df_footbot_positions=Parser.open_pandas_dataframe(filename=file,
                                                              task_name=task_name))

        swarm = Swarm(timesteps=timesteps,
                      swarm=footbots_list,
                      feature_set_features_list=feature_set_feature_list)

        end_time = datetime.now()

        if do_swarm_saving:
            save_swarm(swarm=swarm,
                       file_name=file.split(sep)[-1],
                       task_name=task_name)

        done_files += 1

        print('End time: {}'.format(end_time.strftime('%H:%M %d/%m/%Y')))
        construction_time = end_time - start_time
        delta_times.append(construction_time.total_seconds())

    Parser.save_swarm_construction_time(task_name=task_name,
                                        feature_set_name=feature_set_name,
                                        construction_times=delta_times)


def save_swarm(swarm: Swarm,
               file_name: str,
               task_name: str):
    root = Parser.get_project_root()
    path = join(root, 'cached_files', 'cached_swarms')
    if task_name == 'FLOC':
        path = join(path, 'flocking_swarms', '{}_{}.pkl'.format(task_name, file_name[9:-4]))
        with open(path, 'wb') as output_file:
            pickle.dump(swarm, output_file)
    if task_name == 'HOME':
        path = join(path, 'homing_swarms', '{}_{}.pkl'.format(task_name, file_name[7:-4]))
        with open(path, 'wb') as output_file:
            pickle.dump(swarm, output_file)
    if task_name == 'DISP':
        path = join(path, 'dispersion_swarms',
                    '{}_{}.pkl'.format(task_name, file_name[11:-4]))
        with open(path, 'wb') as output_file:
            pickle.dump(swarm, output_file)
    if task_name == 'FORE':
        path = join(path, 'foraging_swarms', '{}_{}.pkl'.format(task_name, file_name[9:-4]))
        with open(path, 'wb') as output_file:
            pickle.dump(swarm, output_file)
    if task_name == 'WARE':
        path = join(path, 'warehouse_swarms',
                    '{}_{}.pkl'.format(task_name, file_name[10:-4]))
        with open(path, 'wb') as output_file:
            pickle.dump(swarm, output_file)


def build_foraging_swarm(down_sampling: int,
                         feature_set_name: str,
                         feature_set_feature_list: List[str],
                         do_swarm_saving: bool):

    delta_times = []

    neighborhood_radius = Parser.read_neighborhood_radius()
    time_window_size = Parser.read_time_window()

    task_name = 'FORE'
    done_files = 0
    file_list = Parser.read_files_in_log_files_directory(experiment_name=task_name)
    random.shuffle(file_list)
    file_list = file_list[::down_sampling]
    for file in file_list:
        start_time = datetime.now()
        print('Doing file {} out of {}: {}'.format(done_files, len(file_list), file.split(sep)[-1]))
        print('Feature set: {}'.format(feature_set_name))
        print('Start time: {}'.format(start_time.strftime('%H:%M %d/%m/%Y')))
        footbots_list = Parser.create_foraging_swarm(filename=file,
                                                     feature_set_features_list=feature_set_feature_list,
                                                     neighborhood_radius=neighborhood_radius,
                                                     time_window_size=time_window_size)

        timesteps = Parser.retrieve_timesteps_series_from_dataframe(
            df_footbot_positions=Parser.open_pandas_dataframe(filename=file,
                                                              task_name=task_name))

        swarm = Swarm(timesteps=timesteps,
                      swarm=footbots_list,
                      feature_set_features_list=feature_set_feature_list)

        end_time = datetime.now()

        if do_swarm_saving:
            save_swarm(swarm=swarm,
                       task_name='FORE',
                       file_name=file.split(sep)[-1])

        done_files += 1
        print('End time: {}'.format(end_time.strftime('%H:%M %d/%m/%Y')))
        construction_time = end_time - start_time
        delta_times.append(construction_time.total_seconds())

    Parser.save_swarm_construction_time(task_name=task_name,
                                        feature_set_name=feature_set_name,
                                        construction_times=delta_times)


def build_dataset(feature_set_name: str,
                  perform_data_balancing: bool):
    with open('../cached_files/cached_swarms/149_experiments_15_bots.pkl',
              'rb') as input_file:
        experiment_list = pickle.load(input_file)

    print('loaded file')

    down_sampling = Parser.read_timeseries_down_sampling()
    time_window_size = Parser.read_time_window()

    data_wizard = BotDataWizard(
        time_window=time_window_size,
        experiments=experiment_list,
        down_sampling_steps=down_sampling,
        feature_set_name=feature_set_name,
        perform_data_balancing=perform_data_balancing)

    data_wizard_datasets = data_wizard.dataset

    with open('../cached_files/cached_datasets/149exp_15bot_datasets_down_sampled_'
              + str(down_sampling)
              + '_ALL_features_'
              + datetime.now().strftime('%d-%m-%Y_%H-%M') + '.pkl', 'wb') as output_file:
        pickle.dump(data_wizard_datasets, output_file)


def build_feature_set_datasets(task_name: str,
                               experiments_downsampling,
                               delete_useless_bots: bool,
                               useless_bot_deletion_factor: int,
                               perform_data_balancing: bool) -> Dict[str, float]:
    feature_sets_dict = Parser.return_feature_sets_dict()
    timeseries_down_sampling = Parser.read_timeseries_down_sampling()
    time_window_size = Parser.read_time_window()

    pickle_wizard = PickleDataWizard(time_window=time_window_size,
                                     down_sampling_steps=timeseries_down_sampling)
    delta_times_dict = {}
    for main_feature_set_name in feature_sets_dict.keys():
        time_start = datetime.now()
        print('Building feature {} of task {}'.format(main_feature_set_name, task_name))
        pickle_wizard.save_bot_train_test_dataset_specific_swarm(task_name=task_name,
                                                                 feature_set_name=main_feature_set_name,
                                                                 experiments_downsampling=experiments_downsampling,
                                                                 delete_useless_bots=delete_useless_bots,
                                                                 useless_bot_deletion_factor=useless_bot_deletion_factor,
                                                                 perform_data_balancing=perform_data_balancing)
        time_end = datetime.now()
        delta_times_dict[main_feature_set_name] = (time_end - time_start).total_seconds()

    return delta_times_dict


if __name__ == "__main__":
    pass
