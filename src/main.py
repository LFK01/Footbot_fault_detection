import random
import pickle
from src.utils.Parser import Parser
from src.Classes.Swarm import Swarm
from src.utils.SwarmDataWizard import SwarmDataWizard
from src.utils.BotDataWizard import BotDataWizard
from src.utils.FaultDetectionModel import FaultDetectionModel

if __name__ == "__main__":
    # data_wizard = BotDataWizard(
    #    timesteps=SwarmDataWizard.shortest_experiment_timesteps(experiment_list=experiment_list),
    #    time_window=time_window_size,
    #    label_size=1,
    #    experiments=experiment_list,
    #    preprocessing_type='std')

    with open('../cached_objects/bot_data_wizard.pkl', 'rb') as input_file:
        data_wizard = pickle.load(input_file)

    model = FaultDetectionModel(data_wizard=data_wizard)
    model.train_model()

    print(' ')

# ---------------stuff to read file------------
#     random.seed(123)
#
#     neighborhood_radius = Parser.read_neighborhood_radius()
#     time_window_size = Parser.read_time_window()
#
#     experiment_list = []
#     for file in Parser.read_files_in_directory():
#         print(file.split('/')[-1])
#         footbots_list = Parser.create_swarm(filename=file,
#                                             neighborhood_radius=neighborhood_radius,
#                                             time_window_size=time_window_size)
#         # noinspection PyTypeChecker
#         swarm = Swarm(footbots_list)
#         experiment_list.append(swarm)
#
#     random.shuffle(experiment_list)
