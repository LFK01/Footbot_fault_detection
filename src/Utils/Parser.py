import os

import pandas as pd
import numpy as np
from json import load, dump
from os import listdir
from os.path import isfile, join, exists
from pathlib import Path
from src.classes.FootBot import FootBot


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
    def open_parameters_json_file() -> dict:
        # open file
        root = Parser.get_project_root()
        path = join(root, 'txt_files', 'parameters_and_settings.json')
        json_file = open(path)
        data = load(json_file)

        return data

    @staticmethod
    def open_pandas_dataframe(filename: str,
                              task_name: str) -> pd.DataFrame:
        # open file in a pandas dataframe
        root = Parser.get_project_root()
        path = join(root, 'log_files')

        if task_name == 'DISP':
            path = join(path, 'dispersion_log_files', filename)
        elif task_name == 'HOME':
            path = join(path, 'homing_log_files', filename)
        elif task_name == 'FORE':
            path = join(path, 'foraging_log_files', filename)
        elif task_name == 'WARE':
            path = join(path, 'warehouse_log_files', filename)
        elif task_name == 'FLOC':
            path = join(path, 'flocking_log_files', filename)

        df_footbot_positions = pd.read_csv(path,
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
    def retrieve_timesteps_series_from_dataframe(df_footbot_positions: pd.DataFrame) -> np.ndarray:
        # retrieve number of timesteps
        timesteps = df_footbot_positions['timestep'].unique()

        return timesteps

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

        df_footbot_positions = Parser.open_pandas_dataframe(filename=filename,
                                                            task_name='FLOC')

        # retrieve infos
        footbots_unique_ids, number_of_robots, number_of_timesteps, timesteps, all_robots_positions = Parser. \
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

        root = Parser.get_project_root()
        path = join(root, 'log_files', 'foraging_log_files', filename)
        # open file in a pandas dataframe
        df_footbot_positions = pd.read_csv(path)

        # retrieve infos
        footbots_unique_ids, number_of_robots, number_of_timesteps, timesteps, all_robots_positions = Parser. \
            retrieve_dataframe_info(df_footbot_positions)

        for footbot_id in footbots_unique_ids:
            print('parsing bot: ' + str(footbot_id))
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
                                  timesteps=timesteps,
                                  number_of_robots=number_of_robots,
                                  number_of_timesteps=number_of_timesteps,
                                  neighborhood_radius=neighborhood_radius,
                                  time_window_size=time_window_size,
                                  single_robot_positions=all_robots_positions[footbot_id],
                                  all_robots_positions=np.asarray(
                                      [all_robots_positions[key] for key in all_robots_positions.keys()
                                       if int(key) != footbot_id]
                                  ),
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
        task_name: name of the task to read filename from
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
        lstm_length : int
            Value read in the file
        """

        json_data = Parser.open_parameters_json_file()

        return json_data["LSTM_length"]

    @staticmethod
    def read_down_sampling_size() -> int:
        """
        Method to retrieve the down_sampling size in the parameters file.

        Returns
        -------
        down_sampling : int
            Value read in the file
        """

        json_data = Parser.open_parameters_json_file()

        return json_data["Down_sampling"]

    @staticmethod
    def read_area_splits() -> list[int]:
        """
        Method to retrieve the list of area splits in the parameters file.

        Returns
        -------
        area_partitions : list[int]
        """

        json_data = Parser.open_parameters_json_file()

        return json_data["Area_partitions"]

    @staticmethod
    def read_features_set() -> list[str]:
        """
        Method to retrieve the list of features to use in the dataset in the parameters file.

        Returns
        -------
        feature_list: list[str]
        """

        json_data = Parser.open_parameters_json_file()

        return json_data["Features"]

    @staticmethod
    def read_dataset_splittings() -> dict[str, list[float]]:
        """
        Method to retrieve the list of splitting values to use in the building of dataset.

        Returns
        -------
        splitting_dict: dict[str, list[float]]
        """

        json_data = Parser.open_parameters_json_file()

        return json_data["Splitting"]

    @staticmethod
    def read_validation_choice() -> bool:
        """
        Method to check if the validation set has to be built or not.

        Returns
        -------
        validation: bool
        """

        json_data = Parser.open_parameters_json_file()

        return json_data["Validation"]

    @staticmethod
    def read_preprocessing_type() -> str:
        """
        Method to read the preprocessing type.

        Returns
        -------
        preprocessing: str
        """

        json_data = Parser.open_parameters_json_file()

        return json_data["Preprocessing"]

    @staticmethod
    def write_json_file_names(file_names: list[str], task: str) -> None:
        root = Parser.get_project_root()
        path = join(root, 'txt_files')
        json_data = Parser.open_parameters_json_file()
        for index in range(len(file_names)):
            json_data['File Names'][task][str(index)] = file_names[index]
        with open(join(path, 'parameters_and_settings.json'), 'w', encoding='utf-8') as f:
            dump(json_data, f, ensure_ascii=False, indent=4)

    @staticmethod
    def read_files_in_directory(experiment_name: str) -> list:
        root = Parser.get_project_root()
        path = join(root, 'log_files')
        if experiment_name == 'FLOC':
            return [join(path, 'flocking_log_files', f) for f in listdir(join(path, 'flocking_log_files'))
                    if isfile(join(path, 'flocking_log_files', f))]
        elif experiment_name == 'FORE':
            return [join(path, 'foraging_log_files', f) for f in listdir(join(path, 'foraging_log_files'))
                    if isfile(join(path, 'foraging_log_files', f))]
        elif experiment_name == 'DIFF':
            return [join(path, 'diffusion_log_files', f) for f in listdir(join(path, 'diffusion_log_files'))
                    if isfile(join(path, 'diffusion_log_files', f))]
        elif experiment_name == 'DISP':
            return [join(path, 'dispersion_log_files', f) for f in listdir(join(path, 'dispersion_log_files'))
                    if isfile(join(path, 'dispersion_log_files', f))]
        elif experiment_name == 'HOME':
            return [join(path, 'homing_log_files', f) for f in listdir(join(path, 'homing_log_files'))
                    if isfile(join(path, 'homing_log_files', f))]

    @staticmethod
    def remove_ds_store_from_folder(task_name: str):
        root = Parser.get_project_root()
        path = join(root, 'log_files')
        if task_name == 'FLOC':
            path = join(path, 'flocking_log_files')
            if exists(join(path, '.DS_store')):
                os.remove(join(path, '.DS_store'))
                print('REMOVED .DS_store')
        elif task_name == 'FORE':
            path = join(path, 'foraging_log_files')
            if exists(join(path, '.DS_store')):
                os.remove(join(path, '.DS_store'))
                print('REMOVED .DS_store')
        elif task_name == 'DIFF':
            path = join(path, 'diffusion_log_files')
            if exists(join(path, '.DS_store')):
                os.remove(join(path, '.DS_store'))
                print('REMOVED .DS_store')
        elif task_name == 'DISP':
            path = join(path, 'dispersion_log_files')
            if exists(join(path, '.DS_store')):
                os.remove(join(path, '.DS_store'))
                print('REMOVED .DS_store')
        elif task_name == 'HOME':
            path = join(path, 'homing_log_files')
            if exists(join(path, '.DS_store')):
                os.remove(join(path, '.DS_store'))
                print('REMOVED .DS_store')

    @staticmethod
    def sanitize_warehouse_csv_file(task_name: str,
                                    file_number: int):
        filename = Parser.read_filename(task_name=task_name,
                                        file_number=file_number)

        root = Parser.get_project_root()
        path = join(root, filename)

        f = open(path, 'r')

        print('sanitizing file...')
        lines = f.readlines()
        f.close()
        with open(filename, "w") as f:
            for line in lines:
                if "|" not in line.strip("\n"):
                    f.write(line)

        print('finished sanitizing!')


if __name__ == "__main__":
    Parser.remove_ds_store_from_folder(task_name='HOME')
