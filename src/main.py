import random
from src.utils.Parser import Parser
from src.Classes.Swarm import Swarm
from src.utils.DataTools import DataWizard

if __name__ == "__main__":

    random.seed(123)

    neighborhood_radius = Parser.read_neighborhood_radius()
    time_window_size = Parser.read_time_window()

    experiment_list = []
    for file in Parser.read_files_in_directory():
        print(file.split('/')[-1])
        footbots_list = Parser.create_swarm(filename=file,
                                            neighborhood_radius=neighborhood_radius,
                                            time_window_size=time_window_size)
        # noinspection PyTypeChecker
        swarm = Swarm(footbots_list)
        experiment_list.append(swarm)

    random.shuffle(experiment_list)

    data_wizard = DataWizard(timesteps=DataWizard.shortest_experiment_timesteps(experiment_list=experiment_list),
                             time_window=time_window_size,
                             label_size=1,
                             experiments=experiment_list,
                             preprocessing_type='std')

    print(' ')
