import pandas as pd
import numpy as np
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
    def create_swarm(filename: str, neighborhood_radius: float, time_window_size: int) -> list[FootBot]:
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
        swarm : list
            List of all FootBot instances found in the parsed file
        """

        # list to store the robots of the swarm
        footbot_swarm = []

        # open file in a pandas dataframe
        # open file
        try:
            df_footbot_positions = pd.read_csv(filename)
        except FileNotFoundError:
            df_footbot_positions = pd.read_csv('../' + filename)
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

        for footbot_id in footbots_unique_ids:
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
    def read_neighborhood_radius() -> float:
        """
        Method to retrieve the neighborhood_radius in the parameters file.

        Returns
        -------
        neighborhood_radius : float
            Value read in the file
        """

        neighborhood_radius = 0.0

        # open file
        try:
            parameters_files = open('../txt_files/parameters_and_settings')
        except FileNotFoundError:
            parameters_files = open('../../txt_files/parameters_and_settings')

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

        # open file
        try:
            parameters_files = open('../txt_files/parameters_and_settings')
        except FileNotFoundError:
            parameters_files = open('../../txt_files/parameters_and_settings')

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

        # open file
        try:
            parameters_files = open('../txt_files/parameters_and_settings')
        except FileNotFoundError:
            parameters_files = open('../../txt_files/parameters_and_settings')

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
        filename = ""

        # open file
        try:
            parameters_files = open('../txt_files/parameters_and_settings')
        except FileNotFoundError:
            parameters_files = open('../../txt_files/parameters_and_settings')

        # parse file
        for line in parameters_files:
            # fine parameter
            if 'Filename' + str(file_number) in line:
                # retrieve parameter value
                filename = line.split('=')[1].strip()

        # return value
        return filename

    @staticmethod
    def read_files_in_directory() -> list:
        return ['../csv_log_files/' + f for f in listdir('../csv_log_files')
                if isfile(join('../csv_log_files', f))]
