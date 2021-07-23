from src.Classes.AreaPartition import AreaPartition
import numpy as np
from scipy.stats import entropy


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
            [
                [[PosX_1, PosY_1], ..., [PosX_n, PosY_n]]
            ]
        traversed_distance_time_series : np.ndarray
            Array of traversed distances, namely the path covered between each time step. It is initialized with a
            zero value in the first position because every robots starts in a steady state
        direction_time_series: np.ndarray
            Array of the direction vectors composed as [CompX, CompY] for each timestep
        cumulative_traversed_distance : np.ndarray
            Array to store the time series of the cumulative traversed distance considering the timesteps in the time
            window before the considered timesteps. If there aren't enough timesteps to compute the cumulative
            distance, the historical data is assumed to be zero
        swarm_robots_positions : np.ndarray
            list of positions of all the other robots. It is needed to compute the neighbors
            [
                [[PosX_1, PosY_1], ..., [PosX_n, PosY_n]]
                ...
                [[PosX_1, PosY_1], ..., [PosX_n, PosY_n]]
            ]
        swarm_cohesion_time_series: np.ndarray
            A time series of the average distance between this bot and all the others
        neighbors_time_series : np.ndarray
            Array to store the number of neighbors over time

        Methods
        -------
        compute_traversed_space():
            Proceeds on computing the time series of traversed space for each time step

        compute_neighbors():
            Compute the number of robots in the neighborhood for each timestep

        compute_directions():
            Method which computes the distance traversed in each timestep.

        compute_cumulative_traversed_distance():
            Method to compute the cumulative distance traversed from bot in each time step according to time window

        compute_swarm_cohesion(self):
            Method to compute the average distance of the bot from all the other bots for each timestep
        """

    def __init__(self, identifier: int,
                 number_of_robots: int,
                 number_of_timesteps: int,
                 neighborhood_radius: float,
                 time_window_size: int,
                 single_robot_positions: np.ndarray,
                 all_robots_positions: np.ndarray,
                 fault_time_series: np.ndarray,
                 state_time_series: np.ndarray = None,
                 has_food_time_series: np.ndarray = None,
                 total_food_time_series: np.ndarray = None,
                 time_rested_time_series: np.ndarray = None,
                 time_exploring_unsuccessfully_time_series: np.ndarray = None,
                 time_searching_for_nest_time_series: np.ndarray = None):

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
            single_robot_positions: np.ndarray
                Numpy array of the trajectory of the current robot
            all_robots_positions: np.ndarray
                Numpy array of the trajectories of all the other robots
            fault_time_series: np.ndarray
                Numpy array of functioning status of the bot. It will be the target of the learning function
        """

        self.identifier: int = identifier
        self.number_of_robots: int = number_of_robots
        self.number_of_timesteps: int = number_of_timesteps
        self.neighborhood_radius: float = neighborhood_radius
        self.time_window: int = time_window_size

        self.single_robot_positions: np.ndarray = single_robot_positions
        self.traversed_distance_time_series: np.ndarray = np.asarray(0.0)
        self.positions_entropy: np.ndarray = np.asarray(0.0)
        self.direction_time_series: np.ndarray = np.asarray([[0.0, 0.0]])
        self.cumulative_traversed_distance: np.ndarray = np.asarray(0.0)
        self.swarm_robots_positions: np.ndarray = all_robots_positions

        if state_time_series is None:
            self.swarm_robots_positions: np.ndarray = all_robots_positions
            self.swarm_cohesion_time_series: np.ndarray = np.asarray([])
            self.neighbors_time_series: np.ndarray = np.asarray([])

            self.compute_neighbors()
            self.compute_swarm_cohesion()
        else:
            self.state_time_series: np.ndarray = state_time_series
            self.HasFood_time_series: np.ndarray = has_food_time_series
            self.TotalFood_time_series: np.ndarray = total_food_time_series
            self.TimeRested_time_series: np.ndarray = time_rested_time_series
            self.TimeExploringUnsuccessfully_time_series: np.ndarray = time_exploring_unsuccessfully_time_series
            self.TimeSearchingForNest_time_series: np.ndarray = time_searching_for_nest_time_series

        self.distance_from_centroid_time_series: np.ndarray = np.asarray([])
        self.cumulative_distance_from_centroid_time_series: np.ndarray = np.asarray([])
        self.split_2_area_coverage: np.ndarray = np.asarray([])
        self.split_4_area_coverage: np.ndarray = np.asarray([])
        self.split_8_area_coverage: np.ndarray = np.asarray([])

        self.fault_time_series: np.ndarray = fault_time_series

        self.compute_traversed_space()
        self.compute_trajectory_entropy()
        self.compute_directions()
        self.compute_cumulative_traversed_distance()
        self.compute_area_coverage()

    def compute_traversed_space(self) -> None:
        """
        Method which computes the distance traversed in each timestep.
        """
        tmp = []
        previous_position = self.single_robot_positions[0]
        for current_position in self.single_robot_positions[1:]:
            distance_x = previous_position[0] - current_position[0]
            distance_y = previous_position[1] - current_position[1]
            traversed_distance = np.sqrt(distance_x ** 2 + distance_y ** 2)
            tmp.append(traversed_distance)
            previous_position = current_position

        self.traversed_distance_time_series = np.asarray(tmp)

    def compute_trajectory_entropy(self, base=None):
        tmp = []
        for timestep in range(len(self.single_robot_positions)):
            if timestep < self.time_window * 10:
                value, counts = np.unique(self.single_robot_positions[0:timestep],
                                          return_counts=True,
                                          axis=0)
                tmp.append(entropy(counts, base=base))
            else:
                value, counts = np.unique(self.single_robot_positions[timestep - self.time_window * 10:timestep],
                                          return_counts=True,
                                          axis=0)
                tmp.append(entropy(counts, base=base))

        self.positions_entropy = np.asarray(tmp)

    def compute_directions(self) -> None:
        """
        Method which computes the distance traversed in each timestep.
        """
        tmp = []
        previous_position = self.single_robot_positions[0]
        for current_position in self.single_robot_positions[1:]:
            comp_x = previous_position[0] - current_position[0]
            comp_y = previous_position[1] - current_position[1]
            tmp.append([comp_x, comp_y])
            previous_position = current_position

        self.direction_time_series = np.asarray(tmp)

    def compute_neighbors(self) -> None:
        """
        Computes the number of robot within the neighborhood radius at each timestep
        """
        tmp = []
        # get all positions of the first robot and iterate over the timesteps of the positions time series
        for timestep in range(self.swarm_robots_positions.shape[1]):
            # initialize the number of neighbors variable to zero for each time step
            number_of_neighbors = 0
            # iterate over all the remote robots to retrieve their positions at the current timestep
            for remote_robot_positions in self.swarm_robots_positions:
                # compute distance from remote robot
                distance_x = remote_robot_positions[timestep, 0] - self.single_robot_positions[timestep, 0]
                distance_y = remote_robot_positions[timestep, 1] - self.single_robot_positions[timestep, 1]
                distance_between_robots = np.sqrt(distance_x ** 2 + distance_y ** 2)
                # if the distance is below the neighborhood radius then the remote robot is considered as a neighbor
                if distance_between_robots <= self.neighborhood_radius:
                    number_of_neighbors += 1
            # store the collected number of neighbors for the current timestep
            tmp.append(number_of_neighbors)

        self.neighbors_time_series = np.asarray(tmp)

    def compute_cumulative_traversed_distance(self) -> None:
        """
        Method to compute the cumulative distance traversed from bot in each time step according to time window
        """
        tmp = [self.traversed_distance_time_series[0]]
        for i in range(len(self.traversed_distance_time_series))[1:]:
            if i < self.time_window:
                tmp.append(sum(self.traversed_distance_time_series[:i]))
            else:
                tmp.append(sum(self.traversed_distance_time_series[i - self.time_window:i]))
        self.cumulative_traversed_distance = np.asarray(tmp)

    def compute_swarm_cohesion(self) -> None:
        """
        Method to compute the average distance of the bot from all the other bots for each timestep
        """
        tmp = []
        for timestep in range(self.single_robot_positions.shape[0]):
            distances = []
            for remote_bot_positions in self.swarm_robots_positions:
                distance_x = self.single_robot_positions[timestep][0] - remote_bot_positions[timestep][0]
                distance_y = self.single_robot_positions[timestep][1] - remote_bot_positions[timestep][1]
                distances.append(
                    np.sqrt(distance_x ** 2 + distance_y ** 2)
                )
            tmp.append(
                (1 / (self.swarm_robots_positions.shape[1])) * sum(distances)
            )
        self.swarm_cohesion_time_series = np.asarray(tmp)

    def compute_distance_from_centroid(self, trajectory: np.ndarray):
        tmp = []
        for timestep in range(len(trajectory)):
            distance_x, distance_y = trajectory[timestep] - self.single_robot_positions[timestep]
            tmp.append(
                np.sqrt(distance_x ** 2 + distance_y ** 2)
            )
        self.distance_from_centroid_time_series = np.asarray(tmp)

    def compute_cumulative_distance_from_centroid(self):
        tmp = []
        for i in range(len(self.distance_from_centroid_time_series)):
            if i < self.time_window:
                tmp.append(
                    sum(self.distance_from_centroid_time_series[:i]))
            else:
                tmp.append(
                    sum(self.distance_from_centroid_time_series[i - self.time_window:i]))
        self.cumulative_distance_from_centroid_time_series = np.asarray(tmp)

    def compute_area_coverage(self):
        left_bound = np.min(self.swarm_robots_positions[..., 0])
        right_bound = np.max(self.swarm_robots_positions[..., 0])
        low_bound = np.min(self.swarm_robots_positions[..., 1])
        top_bound = np.max(self.swarm_robots_positions[..., 1])

        self.split_2_area_coverage = self.compute_split(split_number=2,
                                                        left_bound=left_bound,
                                                        right_bound=right_bound,
                                                        low_bound=low_bound,
                                                        top_bound=top_bound)
        self.split_4_area_coverage = self.compute_split(split_number=4,
                                                        left_bound=left_bound,
                                                        right_bound=right_bound,
                                                        low_bound=low_bound,
                                                        top_bound=top_bound)
        self.split_8_area_coverage = self.compute_split(split_number=8,
                                                        left_bound=left_bound,
                                                        right_bound=right_bound,
                                                        low_bound=low_bound,
                                                        top_bound=top_bound)

    def compute_split(self,
                      split_number: int,
                      left_bound: float,
                      right_bound: float,
                      low_bound: float,
                      top_bound: float) -> np.ndarray:
        area_coverage = []
        horizontal_splits = [left_bound + abs(right_bound - left_bound) / split_number * repetitions for repetitions in
                             range(split_number+1)]
        vertical_splits = [low_bound + abs(top_bound - low_bound) / split_number * repetitions for repetitions in
                           range(split_number+1)]

        area_partitions = [AreaPartition(left_bound=horizontal_splits[i],
                                         right_bound=horizontal_splits[i + 1],
                                         low_bound=vertical_splits[j],
                                         top_bound=vertical_splits[j + 1])
                           for i in range(len(horizontal_splits) - 1)
                           for j in range(len(vertical_splits) - 1)
                           ]

        positions_index = 0

        while any(not partition.visited for partition in area_partitions) \
                and positions_index < len(self.single_robot_positions):
            position = self.single_robot_positions[positions_index]
            not_visited_partitions = [partition for partition in area_partitions if not partition.visited]
            for partition in not_visited_partitions:
                if (partition.left_bound <= position[0] < partition.right_bound and
                        partition.low_bound <= position[1] < partition.top_bound):
                    partition.visited = True
            area_coverage.append(len([partition for partition in area_partitions if partition.visited])
                                 / len(area_partitions))
            positions_index += 1
        if positions_index < len(self.single_robot_positions):
            area_coverage.append(area_coverage[-1]*(len(self.single_robot_positions)-positions_index))

        return np.asarray(area_coverage)
