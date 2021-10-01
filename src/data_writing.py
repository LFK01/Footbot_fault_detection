import pickle
from datetime import datetime
from src.Utils.Parser import Parser
from src.classes.Swarm import Swarm
from src.Utils.data_utils.BotDataWizard import BotDataWizard
from src.Utils.data_utils.SwarmDataWizard import SwarmDataWizard


def build_swarm():
    neighborhood_radius = Parser.read_neighborhood_radius()
    time_window_size = Parser.read_time_window()

    task_name = 'FLOC'

    experiment_list = []
    for file in Parser.read_files_in_directory(experiment_name=task_name):
        print(file.split('/')[-1])
        footbots_list = Parser.create_flocking_swarm(filename=file,
                                                     neighborhood_radius=neighborhood_radius,
                                                     time_window_size=time_window_size)

        swarm = Swarm(footbots_list)
        experiment_list.append(swarm)

    with open('../cached_files/cached_swarms/experiment_list'
              + datetime.now().strftime('%d-%m-%Y_%H-%M') + '.pkl',
              'wb') as output_file:
        pickle.dump(experiment_list, output_file)

def build_dataset():
    with open('../cached_files/cached_swarms/.pkl',
              'wb') as output_file:
        experiment_list = pickle.load(output_file)

    down_sampling = Parser.read_down_sampling_size()
    timesteps = SwarmDataWizard.shortest_experiment_timesteps(experiment_list=experiment_list)
    time_window_size = Parser.read_time_window()

    data_wizard_datasets = BotDataWizard(
        timesteps=timesteps,
        time_window=time_window_size,
        label_size=1,
        experiments=experiment_list,
        down_sampling_steps=down_sampling,
        preprocessing_type='norm').datasets

    with open('../cached_files/cached_datasets/bot_datasets_down_sampled_' + str(down_sampling) + '_'
              + datetime.now().strftime('%d-%m-%Y_%H-%M') + '.pkl', 'wb') as output_file:
        pickle.dump(data_wizard_datasets, output_file)


if __name__ == "__main__":
    build_swarm()
