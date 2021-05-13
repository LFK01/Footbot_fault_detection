import numpy as np
from src.Classes.FootBot import FootBot


class Swarm:
    """
    Class that collects all the robots in the swarm and computes the features of the cluster
    """

    def __init__(self, swarm: list[FootBot]):
        self.list_of_footbots = swarm
        self.trajectory = self.compute_cluster_trajectory()
        self.traversed_distance_time_series = [0.0]

        self.compute_cluster_speed()
        self.compute_distances_from_centroid()

    def compute_cluster_trajectory(self) -> np.ndarray:
        all_bot_trajectory = np.asarray([bot.single_robot_positions for bot in self.list_of_footbots])
        return np.mean(all_bot_trajectory, axis=0)

    def compute_cluster_speed(self):
        tmp = []
        current_position = self.trajectory[0]
        for next_position in self.trajectory[1:]:
            distance_x = current_position[0] - next_position[0]
            distance_y = current_position[1] - next_position[1]
            traversed_distance = np.sqrt(distance_x**2 + distance_y**2)
            tmp.append(traversed_distance)
        self.traversed_distance_time_series = np.asarray(tmp)

    def compute_distances_from_centroid(self):
        for bot in self.list_of_footbots:
            bot.compute_distance_from_centroid(self.trajectory)
            bot.compute_cumulative_distance_from_centroid()
