import pickle
from datetime import datetime
from src.Utils.Parser import Parser
from src.classes.Swarm import Swarm
from src.Utils.data_utils.data_wizards.BotDataWizard import BotDataWizard
from src.Utils.data_utils.data_wizards.SwarmDataWizard import SwarmDataWizard


def build_swarm():
    neighborhood_radius = Parser.read_neighborhood_radius()
    time_window_size = Parser.read_time_window()

    task_name = 'FLOC'

    experiment_list = []
    done_files = 0
    file_list = Parser.read_files_in_directory(experiment_name=task_name)
    for file in file_list:
        print('Doing file {} out of {}: {}'.format(done_files, len(file_list), file.split('/')[-1]))
        footbots_list = Parser.create_flocking_swarm(filename=file,
                                                     neighborhood_radius=neighborhood_radius,
                                                     time_window_size=time_window_size)

        timesteps = Parser.retrieve_timesteps_series_from_dataframe(
            df_footbot_positions=Parser.open_pandas_dataframe(filename=file,
                                                              task_name=task_name))

        swarm = Swarm(timesteps=timesteps,
                      swarm=footbots_list)

        experiment_list.append(swarm)

        done_files += 1

    with open('../cached_files/cached_swarms/149_experiments'
              + datetime.now().strftime('%d-%m-%Y_%H-%M') +
              '_15_bots.pkl',
              'wb') as output_file:
        pickle.dump(experiment_list, output_file)


def build_dataset():
    with open('../cached_files/cached_swarms/149_experiments_15_bots.pkl',
              'rb') as input_file:
        experiment_list = pickle.load(input_file)

    print('loaded file')

    down_sampling = Parser.read_down_sampling_size()
    timesteps = SwarmDataWizard.shortest_experiment_timesteps(experiment_list=experiment_list)
    time_window_size = Parser.read_time_window()

    data_wizard = BotDataWizard(
        timesteps=timesteps,
        time_window=time_window_size,
        label_size=1,
        experiments=experiment_list,
        down_sampling_steps=down_sampling)

    data_wizard_datasets = data_wizard.datasets

    with open('../cached_files/cached_datasets/149exp_15bot_datasets_down_sampled_'
              + str(down_sampling)
              + '_ALL_features_'
              + datetime.now().strftime('%d-%m-%Y_%H-%M') + '.pkl', 'wb') as output_file:
        pickle.dump(data_wizard_datasets, output_file)


if __name__ == "__main__":
    build_dataset()
