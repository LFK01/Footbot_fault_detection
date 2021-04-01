import math


class FootBot:
    """
        A class used to represent a FootBot.

        Attributes
        ----------
        identifier : int
            integer value to identify the robot among the swarm
        neighborhood_radius : float
            float value to store to maximum distance allowed to identify a robot in the neighborhood
        time_window_size : int
            integer value to identify the maximum number of timesteps to be considered in a window of time
        single_robot_positions : list
            list of coordinates of this instance
        traversed_distance_time_series : list
            List of traversed distances, namely the path covered between each time step. It is initialized with a
            zero value in the first position because every robots starts in a steady state
        cumulative_traversed_distance : list
            List to store the time series of the cumulative traversed distance considering the timesteps in the time
            window before the considered timesteps. If there aren't enough timesteps to compute the cumulative
            distance, the historical data is assumed to be zero
        swarm_robots_positions : list
            list of positions of all the other robots. It is needed to compute the neighbors
        neighbors_time_series : list
            list to store the number of neighbors over time

        Methods
        -------
        add_list_positions(positions_list: list)
            Stores the position_list into single_robot_positions

        compute_traversed_space():
            Proceeds on computing the time series of traversed space for each time step

        add_swarm_robots_positions(all_robots_positions):
            Add all the positions of all the remote robots

        def compute_neighbors():
            Compute the number of robots in the neighborhood for each timestep
        """

    def __init__(self, identifier: int, neighborhood_radius: float, time_window_size: int):
        """
        Constructor method

        Parameters
        ----------
        identifier: int
            Numerical identifier for the robot
        neighborhood_radius: float
            Float radius to define the maximum distance to consider a neighbor. It is retrieved from the
            parameters_and_settings.txt file
        time_window_size: int
            Number of timestep to consider in the window of time. It is retrieved from the
            parameters_and_settings.txt file
        """

        self.identifier = identifier
        self.neighborhood_radius = neighborhood_radius
        self.time_window = time_window_size
        self.single_robot_positions = []
        self.traversed_distance_time_series = [0.0]
        self.cumulative_traversed_distance = [0.0]
        self.swarm_robots_positions = []
        self.neighbors_time_series = []

    def add_list_positions(self, positions_list: list) -> None:
        """
        Method to store the trajectory of the robot.
        Computes the traversed space for each timestep.
        Computes the cumulative traversed distance in the time window.

        Parameters
        ----------
        positions_list : list
            List of tuple coordinates [PosX, PosY]
        """
        self.single_robot_positions.extend(positions_list)
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

    def add_swarm_robots_positions(self, all_robots_positions: list) -> None:
        """
        Store the positions of all the other remote robots.

        Parameters
        ----------
        all_robots_positions : list
            List of all robots trajectories. The trajectory of the current robot has to be left out.
        """
        for i in range(len(all_robots_positions)):
            if i != self.identifier:
                self.swarm_robots_positions.append(all_robots_positions[i])
        self.compute_neighbors()

    def compute_neighbors(self) -> None:
        """
        Computes the number of robot within the neighborhood radius at each timestep
        """
        # get all positions of the first robot and iterate over the timesteps of the positions time series
        for timestep in range(len(self.swarm_robots_positions[0])):
            # initialize the number of neighbors variable to zero for each time step
            number_of_neighbors = 0
            # iterate over all the remote robots to retrieve their positions at the current timestep
            for remote_robot_positions in self.swarm_robots_positions:
                # compute distance from remote robot
                distance_x = remote_robot_positions[timestep][0] - self.single_robot_positions[timestep][0]
                distance_y = remote_robot_positions[timestep][1] - self.single_robot_positions[timestep][1]
                distance_between_robots = math.sqrt(distance_x**2 + distance_y**2)
                # if the distance is below the neighborhood radius then the remote robot is considered as a neighbor
                if distance_between_robots <= self.neighborhood_radius:
                    number_of_neighbors += 1
            # store the collected number of neighbors for the current timestep
            self.neighbors_time_series.append(number_of_neighbors)
