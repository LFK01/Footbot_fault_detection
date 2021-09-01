import pickle
from src.Utils.Parser import Parser
from src.Classes.Swarm import Swarm
from src.Utils.data_utils.BotDataWizard import BotDataWizard
from src.Utils.data_utils.SwarmDataWizard import SwarmDataWizard

if __name__ == "__main__":
    neighborhood_radius = Parser.read_neighborhood_radius()
    time_window_size = Parser.read_time_window()

    experiment_list = []
    for file in Parser.read_files_in_directory():
        print(file.split('/')[-1])
        footbots_list = Parser.create_flocking_swarm(filename=file,
                                                     neighborhood_radius=neighborhood_radius,
                                                     time_window_size=time_window_size)

        swarm = Swarm(footbots_list)
        experiment_list.append(swarm)

    timesteps = SwarmDataWizard.shortest_experiment_timesteps(experiment_list=experiment_list)

    data_wizard_datasets = BotDataWizard(
        timesteps=timesteps,
        time_window=time_window_size,
        label_size=1,
        experiments=experiment_list,
        down_sampling_steps=10,
        preprocessing_type='norm').datasets

    with open('../cached_objects/bot_datasets_down_sampled.pkl', 'wb') as output_file:
        pickle.dump(data_wizard_datasets, output_file)
