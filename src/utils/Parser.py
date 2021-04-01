import pandas as pd
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

    all_robots_positions = []

    def __init__(self):
        """
        Constructor empty
        """

        pass

    @staticmethod
    def create_swarm(filename: str, neighborhood_radius: float) -> list:
        """

        Parameters
        ----------
        filename : str
            String which specifies the name of the file to read
        neighborhood_radius : float
            Float value which specifies the maximum distance allowed to identify a robot in the neighborhood

        Returns
        -------
        swarm : list
            List of all FootBot instances found in the paprsed file
        """
        # list to store the robots of the swarm
        footbot_swarm = []
        # list to store all the positions of the grouped by bot
        all_robots_positions = []

        # open file in a pandas dataframe
        df_footbot_positions = pd.read_csv('../csv_log_files/' + filename)
        # retrieve all the ids of the bot
        footbots_unique_ids = df_footbot_positions['ID'].unique()

        for footbot_id in footbots_unique_ids:
            # create new FootBot instance
            new_footbot = FootBot(footbot_id, neighborhood_radius)

            # retrieve positions of the current robots based on its ID
            positions = df_footbot_positions[df_footbot_positions['ID'] == footbot_id][['PosX', 'PosY']]
            # save the position of the robot on the istance of the robot
            new_footbot.add_list_positions(positions.values.tolist())

            # Save the positions of the robot on the list of all the positions of the robots.
            # Now the positions are grouped by robot ID and not by timestep as in the csv file
            all_robots_positions.append(positions.values.tolist())

            # save new FootBot instance in the swarm
            footbot_swarm.append(new_footbot)

        # save the positions of all the remote robots in the all the robots instance
        for robot in footbot_swarm:
            robot.add_swarm_robots_positions(all_robots_positions)

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
        parameters_files = open('../txt_files/parameters_and_settings')

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
        parameters_files = open('../txt_files/parameters_and_settings')

        # parse file
        for line in parameters_files:
            # fine parameter
            if 'TIME_WINDOW' in line:
                # retrieve parameter value
                time_window = int(line.split('=')[1].replace(' ', ''))

        # return value
        return time_window
