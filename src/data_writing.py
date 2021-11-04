import pickle
import random
from datetime import datetime
from os import listdir, sep
from os.path import join
from src.Utils.Parser import Parser
from src.classes.Swarm import Swarm
from src.Utils.data_utils.data_wizards.DataWizard import DataWizard
from src.Utils.data_utils.data_wizards.BotDataWizard import BotDataWizard
from src.Utils.data_utils.data_wizards.PickleDataWizard import PickleDataWizard


def build_swarm_no_foraging_stats(task_name: str,
                                  experiments_number_down_sampling: int = 1):
    neighborhood_radius = Parser.read_neighborhood_radius()
    time_window_size = Parser.read_time_window()

    file_list = Parser.read_files_in_log_files_directory(experiment_name=task_name)
    random.seed(Parser.read_seed())
    done_files = 1
    random.shuffle(file_list)
    file_list = file_list[::experiments_number_down_sampling]
    for file in file_list:
        print('Doing file {} out of {}: {}'.format(done_files, len(file_list), file.split(sep)[-1]))
        footbots_list = Parser.create_generic_swarm(task_name=task_name,
                                                    filename=file,
                                                    neighborhood_radius=neighborhood_radius,
                                                    time_window_size=time_window_size)

        timesteps = Parser.retrieve_timesteps_series_from_dataframe(
            df_footbot_positions=Parser.open_pandas_dataframe(filename=file,
                                                              task_name=task_name))

        swarm = Swarm(timesteps=timesteps,
                      swarm=footbots_list)

        save_swarm(swarm=swarm,
                   file_name=file.split(sep)[-1],
                   task_name=task_name)
        done_files += 1


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


def build_foraging_swarm(down_sampling: int):
    neighborhood_radius = Parser.read_neighborhood_radius()
    time_window_size = Parser.read_time_window()

    task_name = 'FORE'
    done_files = 0
    file_list = Parser.read_files_in_log_files_directory(experiment_name=task_name)
    random.shuffle(file_list)
    file_list = file_list[::down_sampling]
    for file in file_list:
        print('Doing file {} out of {}: {}'.format(done_files, len(file_list), file.split(sep)[-1]))
        footbots_list = Parser.create_foraging_swarm(filename=file,
                                                     neighborhood_radius=neighborhood_radius,
                                                     time_window_size=time_window_size)

        timesteps = Parser.retrieve_timesteps_series_from_dataframe(
            df_footbot_positions=Parser.open_pandas_dataframe(filename=file,
                                                              task_name=task_name))

        swarm = Swarm(timesteps=timesteps,
                      swarm=footbots_list)

        save_swarm(swarm=swarm,
                   task_name='FORE',
                   file_name=file.split(sep)[-1])

        done_files += 1


def build_dataset(feature_set_number: int,
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
        feature_set_number=feature_set_number,
        perform_data_balancing=perform_data_balancing)

    data_wizard_datasets = data_wizard.dataset

    with open('../cached_files/cached_datasets/149exp_15bot_datasets_down_sampled_'
              + str(down_sampling)
              + '_ALL_features_'
              + datetime.now().strftime('%d-%m-%Y_%H-%M') + '.pkl', 'wb') as output_file:
        pickle.dump(data_wizard_datasets, output_file)


def build_feature_set_datasets(task_name: str,
                               experiments_downsampling,
                               useless_bot_deletion_factor: int,
                               perform_data_balancing: bool):
    f_numbers = [1, 2, 3]
    timeseries_down_sampling = Parser.read_timeseries_down_sampling()
    time_window_size = Parser.read_time_window()

    pickle_wizard = PickleDataWizard(time_window=time_window_size,
                                     down_sampling_steps=timeseries_down_sampling)

    for main_feature_set_number in f_numbers:
        print('Building feature set {} of task {}'.format(main_feature_set_number, task_name))
        pickle_wizard.save_bot_train_test_dataset_specific_swarm(task_name=task_name,
                                                                 feature_set_number=main_feature_set_number,
                                                                 experiments_downsampling=experiments_downsampling,
                                                                 useless_bot_deletion_factor=useless_bot_deletion_factor,
                                                                 perform_data_balancing=perform_data_balancing)


if __name__ == "__main__":
    build_feature_set_datasets(task_name='FLOC',
                               experiments_downsampling=1,
                               useless_bot_deletion_factor=4,
                               perform_data_balancing=True)
