import random
import pickle
from src.utils.Parser import Parser
from src.Classes.Swarm import Swarm
from src.utils.FaultDetectionModel import FaultDetectionModel
from src.utils.data_utils.SwarmDataWizard import SwarmDataWizard
from src.utils.data_utils.BotDataWizard import BotDataWizard
from src.utils.data_utils.BotDataset import BotDataset


if __name__ == "__main__":
    # neighborhood_radius = Parser.read_neighborhood_radius()
    # time_window_size = Parser.read_time_window()
    #
    # experiment_list = []
    # for file in Parser.read_files_in_directory():
    #     print(file.split('/')[-1])
    #     footbots_list = Parser.create_swarm(filename=file,
    #                                         neighborhood_radius=neighborhood_radius,
    #                                         time_window_size=time_window_size)
    #     # noinspection PyTypeChecker
    #     swarm = Swarm(footbots_list)
    #     experiment_list.append(swarm)
    #
    # data_wizard_datasets = BotDataWizard(
    #     timesteps=SwarmDataWizard.shortest_experiment_timesteps(experiment_list=experiment_list),
    #     time_window=time_window_size,
    #     label_size=1,
    #     experiments=experiment_list,
    #     preprocessing_type='norm').datasets
    #
    # with open('../cached_objects/bot_data_wizard_normalized.pkl', 'wb') as output_file:
    #     pickle.dump(data_wizard_datasets, output_file)

    with open('../cached_objects/bot_data_wizard_normalized.pkl', 'rb') as input_file:
        data_wizard_datasets: list[BotDataset] = pickle.load(input_file)

    model = FaultDetectionModel(bot_datasets=data_wizard_datasets)
    model.train_model()

    print(' ')

# ---------------stuff to read file------------
