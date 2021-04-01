import numpy as np
import math


class FootBot:
    """
        A class used to represent a FootBot.

        Attributes
        ----------
        identifier : int
            integer value to identify the robot among the swarm
        number_of_robots : int
            Number of robots in the swarm
        number_of_timesteps : int
            Length of the time series
        neighborhood_radius : float
            float value to store to maximum distance allowed to identify a robot in the neighborhood
        time_window_size : int
            integer value to identify the maximum number of timesteps to be considered in a window of time
        single_robot_positions : np.ndarray
            list of coordinates of this instance
        traversed_distance_time_series : np.ndarray
            List of traversed distances, namely the path covered between each time step. It is initialized with a
            zero value in the first position because every robots starts in a steady state
        cumulative_traversed_distance : np.ndarray
            List to store the time series of the cumulative traversed distance considering the timesteps in the time
            window before the considered timesteps. If there aren't enough timesteps to compute the cumulative
            distance, the historical data is assumed to be zero
        swarm_robots_positions : np.ndarray
            list of positions of all the other robots. It is needed to compute the neighbors
        neighbors_time_series : np.ndarray
            list to store the number of neighbors over time

        Methods
        -------
        add_list_positions(positions_list: np.ndarray)
            Stores the position_list into single_robot_positions

        compute_traversed_space():
            Proceeds on computing the time series of traversed space for each time step

        add_swarm_robots_positions(all_robots_positions: np.ndarray):
            Add all the positions of all the remote robots

        def compute_neighbors():
            Compute the number of robots in the neighborhood for each timestep
        """

    def __init__(self, identifier: int, number_of_robots: int, number_of_timesteps: int,
                 neighborhood_radius: float, time_window_size: int):
        """
        Constructor method

        Parameters
        ----------
        identifier: int
            Numerical identifier for the robot
        number_of_robots: int
            Number of robots in the swarm
        number_of_timesteps: int
            Length of the time series
        neighborhood_radius: float
            Float radius to define the maximum distance to consider a neighbor. It is retrieved from the
            parameters_and_settings.txt file
        time_window_size: int
            Number of timestep to consider in the window of time. It is retrieved from the
            parameters_and_settings.txt file
        """

        self.identifier = identifier
        self.number_of_robots = number_of_robots
        self.number_of_timesteps = number_of_timesteps
        self.neighborhood_radius = neighborhood_radius
        self.time_window = time_window_size
        self.single_robot_positions = np.asarray([])
        self.traversed_distance_time_series = [0.0]
        self.cumulative_traversed_distance = [0.0]
        self.swarm_robots_positions = np.asarray([])
        self.neighbors_time_series = []

    def add_array_positions(self, positions_list: np.ndarray) -> None:
        """
        Method to store the trajectory of the robot.
        Computes the traversed space for each timestep.
        Computes the cumulative traversed distance in the time window.

        Parameters
        ----------
        positions_list : np.ndarray
            numpy array of tuple coordinates [PosX, PosY]
        """
        self.single_robot_positions = positions_list
        self.compute_traversed_space()

    def compute_traversed_space(self) -> None:
        """
        Method which computes the distance traversed in each timestep.
        """
        previous_position = self.single_robot_positions[0]
        for current_position in self.single_robot_positions[1:]:
            distance_x = previous_position[0] - current_position[0]
            distance_y = previous_position[1] - current_position[1]
            traversed_distance = math.sqrt(distance_x**2 + distance_y**2)
            self.traversed_distance_time_series.append(traversed_distance)
            previous_position = current_position

        self.traversed_distance_time_series = np.asarray(self.traversed_distance_time_series)

    def add_swarm_robots_positions(self, all_robots_positions: np.ndarray) -> None:
        """
        Store the positions of all the other remote robots.

        Parameters
        ----------
        all_robots_positions : np.ndarray
            numpy array of all robots trajectories. The trajectory of the current robot has to be left out.
        """
        self.swarm_robots_positions = np.delete(all_robots_positions, self.identifier, axis=0)
        self.compute_neighbors()

    def compute_neighbors(self) -> None:
        """
        Computes the number of robot within the neighborhood radius at each timestep
        """
        # get all positions of the first robot and iterate over the timesteps of the positions time series
        for timestep in range(self.swarm_robots_positions.shape[1]):
            # initialize the number of neighbors variable to zero for each time step
            number_of_neighbors = 0
            # iterate over all the remote robots to retrieve their positions at the current timestep
            for remote_robot_positions in self.swarm_robots_positions:
                # compute distance from remote robot
                distance_x = remote_robot_positions[timestep, 0] - self.single_robot_positions[timestep, 0]
                distance_y = remote_robot_positions[timestep, 1] - self.single_robot_positions[timestep, 1]
                distance_between_robots = math.sqrt(distance_x**2 + distance_y**2)
                # if the distance is below the neighborhood radius then the remote robot is considered as a neighbor
                if distance_between_robots <= self.neighborhood_radius:
                    number_of_neighbors += 1
            # store the collected number of neighbors for the current timestep
            self.neighbors_time_series.append(number_of_neighbors)

        self.neighbors_time_series = np.asarray(self.neighbors_time_series)
