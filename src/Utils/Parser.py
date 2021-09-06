import pandas as pd
import numpy as np
import json
from os import listdir
from os.path import isfile, join
from pathlib import Path
from src.Classes.FootBot import FootBot


class Parser:
    """
    Class used to parse files.
    """

    def __init__(self):
        """
        Constructor empty
        """

        pass

    @staticmethod
    def get_project_root() -> Path:
        return Path(__file__).parent.parent.parent

    @staticmethod
    def open_parameters_json_file():
        # open file
        try:
            json_file = open('../txt_files/parameters_and_settings')
            data = json.load(json_file)
        except FileNotFoundError:
            try:
                json_file = open('../../txt_files/parameters_and_settings')
                data = json.load(json_file)
            except FileNotFoundError:
                json_file = open('txt_files/parameters_and_settings')
                data = json.load(json_file)

        return data

    @staticmethod
    def open_pandas_dataframe(filename: str) -> pd.DataFrame:
        # open file in a pandas dataframe
        try:
            df_footbot_positions = pd.read_csv(filename,
                                               dtype={'timestep': float,
                                                      'ID': int,
                                                      'PosX': float,
                                                      'PosY': float,
                                                      'Fault': bool})
        except FileNotFoundError:
            df_footbot_positions = pd.read_csv('../' + filename,
                                               dtype={'timestep': float,
                                                      'ID': int,
                                                      'PosX': float,
                                                      'PosY': float,
                                                      'Fault': bool})

        return df_footbot_positions

    @staticmethod
    def retrieve_dataframe_info(df_footbot_positions: pd.DataFrame) -> tuple[list[int], int, int, np.ndarray, dict]:
        # retrieve all the ids of the bot
        footbots_unique_ids = df_footbot_positions['ID'].unique().astype(int)
        number_of_robots = len(footbots_unique_ids)
        number_of_timesteps = len(df_footbot_positions['timestep'].unique())
        timesteps = df_footbot_positions['timestep'].unique()

        # list to store all the positions of the swarm grouped by bot
        all_robots_positions = {}
        for footbot_id in footbots_unique_ids:
            # retrieve positions of the current robots based on its ID
            positions = df_footbot_positions[df_footbot_positions['ID'] == footbot_id][['PosX', 'PosY']]
            positions = positions.to_numpy()

            # Save the positions of the robot on the list of all the positions of the robots.
            # Now the positions are grouped by robot ID and not by timestep as in the csv file
            all_robots_positions[footbot_id] = positions

        return footbots_unique_ids, number_of_robots, number_of_timesteps, np.asarray(timesteps), all_robots_positions

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
        footbots_unique_ids, number_of_robots, number_of_timesteps, timesteps, all_robots_positions = Parser.\
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
                                  single_robot_positions=np.asarray(all_robots_positions[footbot_id]),
                                  timesteps=timesteps,
                                  all_robots_positions=np.asarray(
                                      [all_robots_positions[key] for key in all_robots_positions.keys()
                                       if int(key) != footbot_id]
                                  ),
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
        footbots_unique_ids, number_of_robots, number_of_timesteps, timesteps, all_robots_positions = Parser. \
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

        json_data = Parser.open_parameters_json_file()

        return json_data["NEIGHBORHOOD_RADIUS"]

    @staticmethod
    def read_time_window() -> int:
        """
        Method to retrieve the time_window in the parameters file.

        Returns
        -------
        time_window : int
            Value read in the file
        """

        json_data = Parser.open_parameters_json_file()

        return json_data["TIME_WINDOW"]

    @staticmethod
    def read_seed() -> int:
        """
        Method to retrieve the seed in the parameters file.

        Returns
        -------
        seed : int
            Value read in the file
        """

        json_data = Parser.open_parameters_json_file()

        return json_data["SEED"]

    @staticmethod
    def read_filename(task_name: str,
                      file_number: int) -> str:
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
        json_data = Parser.open_parameters_json_file()

        return json_data["File Names"][task_name][str(file_number)]

    @staticmethod
    def read_lstm_length() -> int:
        """
        Method to retrieve the LSTM length in the parameters file.

        Returns
        -------
        seed : int
            Value read in the file
        """

        json_data = Parser.open_parameters_json_file()

        return json_data["LSTM_length"]

    @staticmethod
    def read_area_splits() -> list:
        """
        Method to retrieve the LSTM length in the parameters file.

        Returns
        -------
        seed : int
            Value read in the file
        """

        json_data = Parser.open_parameters_json_file()

        return json_data["Area_partitions"]

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
    def sanitize_warehouse_csv_file(task_name: str,
                                    file_number: int):
        filename = Parser.read_filename(task_name=task_name,
                                        file_number=file_number)
        try:
            f = open(filename, "r")
        except FileNotFoundError:
            filename = '../' + filename
            f = open(filename, "r")

        print('sanitizing file...')
        lines = f.readlines()
        f.close()
        with open(filename, "w") as f:
            for line in lines:
                if "|" not in line.strip("\n"):
                    f.write(line)

        print('finished sanitizing!')


if __name__ == "__main__":
    task_name = "WARE"
    file_number = 9
    Parser.sanitize_warehouse_csv_file(task_name=task_name,
                                       file_number=file_number)
