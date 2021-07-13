import pandas as pd
import numpy as np
import tqdm
from os import listdir
from os.path import isfile, join
from src.Classes.FootBot import FootBot


class Parser:
    """
    Class used to parse files.

    Attributes
    ----------
    all_robots_positions : list
        Numpy array of the trajectories of all the robots composed as:
        [
            [[PosX_1, PosY_1], ..., [PosX_n, PosY_n]]
            ...
            [[PosX_1, PosY_1], ..., [PosX_n, PosY_n]]
        ]
    """

    def __init__(self):
        """
        Constructor empty
        """

        pass

    @staticmethod
    def open_parameters_file():
        # open file
        try:
            parameters_files = open('../txt_files/parameters_and_settings')
        except FileNotFoundError:
            try:
                parameters_files = open('../../txt_files/parameters_and_settings')
            except FileNotFoundError:
                parameters_files = open('txt_files/parameters_and_settings')

        return parameters_files

    @staticmethod
    def open_pandas_dataframe(filename: str) -> pd.DataFrame:
        # open file in a pandas dataframe
        try:
            df_footbot_positions = pd.read_csv(filename)
        except FileNotFoundError:
            df_footbot_positions = pd.read_csv('../' + filename)

        return df_footbot_positions

    @staticmethod
    def retrieve_dataframe_info(df_footbot_positions: pd.DataFrame) -> tuple[list[int], int, int, np.ndarray]:
        # retrieve all the ids of the bot
        footbots_unique_ids = df_footbot_positions['ID'].unique()
        number_of_robots = len(footbots_unique_ids)
        number_of_timesteps = len(df_footbot_positions['timestep'].unique())

        # list to store all the positions of the swarm grouped by bot
        all_robots_positions = []
        for footbot_id in footbots_unique_ids:
            # retrieve positions of the current robots based on its ID
            positions = df_footbot_positions[df_footbot_positions['ID'] == footbot_id][['PosX', 'PosY']]
            positions = positions.to_numpy()

            # Save the positions of the robot on the list of all the positions of the robots.
            # Now the positions are grouped by robot ID and not by timestep as in the csv file
            all_robots_positions.append(positions)

        # create numpy array
        all_robots_positions = np.asarray(all_robots_positions)

        return footbots_unique_ids, number_of_robots, number_of_timesteps, all_robots_positions

    @staticmethod
    def create_flocking_swarm(filename: str, neighborhood_radius: float, time_window_size: int) -> list[FootBot]:
        """
        Method to parse the positions file and return the list of footbots.

        Parameters
        ----------
        filename : str
            String which specifies the name of the file to read
        neighborhood_radius : float
            Float value which specifies the maximum distance allowed to identify a robot in the neighborhood
        time_window_size : int
            Integer value to identify the maximum number of timesteps to be considered in a window of time
        Returns
        -------
        swarm : list[FootBot]
            List of all FootBot instances found in the parsed file
        """

        # list to store the robots of the swarm
        footbot_swarm = []

        df_footbot_positions = Parser.open_pandas_dataframe(filename=filename)

        # retrieve infos
        footbots_unique_ids, number_of_robots, number_of_timesteps, all_robots_positions = Parser.\
            retrieve_dataframe_info(df_footbot_positions)

        for footbot_id in footbots_unique_ids:
            print('parsing bot: ' + str(footbot_id))
            # retrieve faults of the current robots based on its ID
            faults = df_footbot_positions[df_footbot_positions['ID'] == footbot_id]['Fault']
            faults = faults.to_numpy(dtype=bool)

            # create new FootBot instance
            new_footbot = FootBot(identifier=footbot_id,
                                  number_of_robots=number_of_robots,
                                  number_of_timesteps=number_of_timesteps,
                                  neighborhood_radius=neighborhood_radius,
                                  time_window_size=time_window_size,
                                  single_robot_positions=all_robots_positions[footbot_id],
                                  all_robots_positions=np.delete(all_robots_positions, footbot_id, axis=0),
                                  fault_time_series=faults)

            # save new FootBot instance in the swarm
            footbot_swarm.append(new_footbot)

        return footbot_swarm

    @staticmethod
    def create_foraging_swarm(filename: str, neighborhood_radius: float, time_window_size: int) -> list[FootBot]:
        # list to store the robots of the swarm
        footbot_swarm = []

        # open file in a pandas dataframe
        try:
            df_footbot_positions = pd.read_csv(filename)
        except FileNotFoundError:
            df_footbot_positions = pd.read_csv('../' + filename)

        # retrieve infos
        footbots_unique_ids, number_of_robots, number_of_timesteps, all_robots_positions = Parser. \
            retrieve_dataframe_info(df_footbot_positions)

        for footbot_id in footbots_unique_ids:
            # retrieve faults of the current robots based on its ID
            faults = df_footbot_positions[df_footbot_positions['ID'] == footbot_id]['Fault'].to_numpy(dtype=bool)

            # retrieve states of the current robots based on its ID
            states = df_footbot_positions[df_footbot_positions['ID'] == footbot_id]['State'].to_numpy(dtype=int)

            # retrieve HasFood information of the current robots based on its ID
            food = df_footbot_positions[df_footbot_positions['ID'] == footbot_id]['HasFood'].to_numpy(dtype=bool)

            # retrieve TotalFood information of the current robots based on its ID
            total_food = df_footbot_positions[df_footbot_positions['ID'] == footbot_id]['TotalFood'].to_numpy(dtype=int)

            # retrieve Time_Rested information of the current robots based on its ID
            time_rested = df_footbot_positions[df_footbot_positions['ID'] == footbot_id][
                'TimeRested'].to_numpy(dtype=int)

            # retrieve Time_Exploring_Unsuccessfully information of the current robots based on its ID
            exploration_time = df_footbot_positions[df_footbot_positions['ID'] == footbot_id][
                'TimeExploringUnsuccessfully'].to_numpy(dtype=int)

            # retrieve Time_Exploring_Unsuccessfully information of the current robots based on its ID
            searching_space_in_nest_time = df_footbot_positions[df_footbot_positions['ID'] == footbot_id][
                'TimeSearchingForNest'].to_numpy(dtype=int)

            # create new FootBot instance
            new_footbot = FootBot(identifier=footbot_id,
                                  number_of_robots=number_of_robots,
                                  number_of_timesteps=number_of_timesteps,
                                  neighborhood_radius=neighborhood_radius,
                                  time_window_size=time_window_size,
                                  single_robot_positions=all_robots_positions[footbot_id],
                                  all_robots_positions=np.delete(all_robots_positions, footbot_id, axis=0),
                                  state_time_series=states,
                                  has_food_time_series=food,
                                  total_food_time_series=total_food,
                                  time_rested_time_series=time_rested,
                                  time_exploring_unsuccessfully_time_series=exploration_time,
                                  time_searching_for_nest_time_series=searching_space_in_nest_time,
                                  fault_time_series=faults)

            # save new FootBot instance in the swarm
            footbot_swarm.append(new_footbot)

        return footbot_swarm

    @staticmethod
    def read_neighborhood_radius() -> float:
        """
        Method to retrieve the neighborhood_radius in the parameters file.

        Returns
        -------
        neighborhood_radius : float
            Value read in the file
        """

        neighborhood_radius = 0.0

        parameters_files = Parser.open_parameters_file()

        # parse file
        for line in parameters_files:
            # fine parameter
            if 'NEIGHBORHOOD_RADIUS' in line:
                # retrieve parameter value
                neighborhood_radius = float(line.split('=')[1].replace(' ', ''))

        # return value
        return neighborhood_radius

    @staticmethod
    def read_time_window() -> int:
        """
        Method to retrieve the time_window in the parameters file.

        Returns
        -------
        time_window : int
            Value read in the file
        """

        time_window = 0

        parameters_files = Parser.open_parameters_file()

        # parse file
        for line in parameters_files:
            # fine parameter
            if 'TIME_WINDOW' in line:
                # retrieve parameter value
                time_window = int(line.split('=')[1].replace(' ', ''))

        # return value
        return time_window

    @staticmethod
    def read_seed() -> int:
        """
        Method to retrieve the seed in the parameters file.

        Returns
        -------
        seed : int
            Value read in the file
        """

        seed = 0

        parameters_files = Parser.open_parameters_file()

        # parse file
        for line in parameters_files:
            # fine parameter
            if 'SEED' in line:
                # retrieve parameter value
                seed = int(line.split('=')[1].replace(' ', ''))

        # return value
        return seed

    @staticmethod
    def read_filename(file_number: int) -> str:
        """
        Method to read the name of csv files
        Parameters
        ----------
        file_number: int
            Number of filename in parameters_and_settings.txt
        Returns
        -------
        filename: str
            String of the file name
        """
        filename = ''

        parameters_files = Parser.open_parameters_file()

        # parse file
        for line in parameters_files:
            # fine parameter
            if 'Filename' + str(file_number) in line:
                # retrieve parameter value
                filename = line.split('=')[1].strip()

        # return value
        return filename

    @staticmethod
    def read_lstm_length() -> int:
        """
        Method to retrieve the LSTM length in the parameters file.

        Returns
        -------
        seed : int
            Value read in the file
        """

        lstm_length = 0

        parameters_files = Parser.open_parameters_file()

        # parse file
        for line in parameters_files:
            # fine parameter
            if 'LSTM_length' in line:
                # retrieve parameter value
                lstm_length = int(line.split('=')[1].replace(' ', ''))

        # return value
        return lstm_length

    @staticmethod
    def read_files_in_directory(experiment_name: str) -> list:
        if experiment_name == 'flocking':
            return ['../flocking_log_files/' + f for f in listdir('../flocking_log_files')
                    if isfile(join('../flocking_log_files', f))]
        elif experiment_name == 'foraging':
            return ['../foraging_log_files/' + f for f in listdir('../foraging_log_files')
                    if isfile(join('../foraging_log_files', f))]
        elif experiment_name == 'diffusion':
            return ['../diffusion_log_files/' + f for f in listdir('../diffusion_log_files')
                    if isfile(join('../diffusion_log_files', f))]
        elif experiment_name == 'dispersion':
            return ['../dispersion_log_files/' + f for f in listdir('../dispersion_log_files')
                    if isfile(join('../dispersion_log_files', f))]
        elif experiment_name == 'homing':
            return ['../homing_log_files/' + f for f in listdir('../homing_log_files')
                    if isfile(join('../homing_log_files', f))]
        else:
            raise FileNotFoundError

    @staticmethod
    def clean_heat_file():
        # clean file
        input_file = 'diffusion_log_files/locationspolled.heat'
        output_file = '../../diffusion_log_files/locationspolled.csv'

        try:
            in_file = open(input_file)
        except FileNotFoundError:
            in_file = open('../../' + input_file)

        for line in tqdm.tqdm(in_file):
            if '#<PolledLocation>:' in line:
                line = line.replace('#<PolledLocation>:', '')
            line = line.replace(';', ',')
            with open(output_file, 'a+') as o_file:
                o_file.write(line)

        in_file.close()
        o_file.close()

    @staticmethod
    def add_bot_id():
        csv_file = '../../diffusion_log_files/locationspolled.csv'

        # open file in a pandas dataframe
        try:
            bot_dataframe = pd.read_csv(csv_file)
        except FileNotFoundError:
            bot_dataframe = pd.read_csv('../../' + csv_file)

        unique_timesteps = bot_dataframe['TimeStamp'].unique()
        first_timestep = unique_timesteps[0]
        bot_number = len(bot_dataframe.loc[bot_dataframe['TimeStamp'] == first_timestep])

        faults = [0] * bot_number
        faults[1] = 1
        faults = faults * len(unique_timesteps)

        bot_IDs = list(range(0, bot_number))
        bot_IDs = bot_IDs * len(unique_timesteps)

        bot_dataframe.insert(0, 'bot_ID', bot_IDs)
        bot_dataframe.insert(len(bot_dataframe.columns), 'Fault', faults)

        bot_dataframe.to_csv(csv_file)


if __name__ == "__main__":
    csv_file = '../../warehouse_log_files/locationspolled.csv'
    df = pd.read_csv(csv_file)
    print(df['BotTask'].unique())
