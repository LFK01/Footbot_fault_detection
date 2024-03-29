import numpy as np
from typing import List

from src.Utils.Parser import Parser
from src.classes.FootBot import FootBot
from src.classes.AreaPartition import AreaPartition


class Swarm:
    """
    Class that collects all the robots in the swarm and computes the features of the cluster
    """

    def __init__(self,
                 feature_set_features_list: List[str],
                 timesteps: np.ndarray,
                 swarm: List[FootBot]):
        self.timesteps = timesteps
        self.list_of_footbots = swarm
        self.trajectory = self.compute_cluster_trajectory()
        self.speed_time_series = [0.0]
        self.area_partitions = self.compute_area_partitions()
        if 'global_features' in feature_set_features_list:
            self.area_coverage = self.compute_area_coverage()

        if 'swarm_speed' in feature_set_features_list:
            self.compute_cluster_speed()
        if 'distance_from_centroid_time_series' in feature_set_features_list:
            self.compute_distances_from_centroid()
        if 'area_coverage' in feature_set_features_list:
            self.compute_single_bots_area_coverage()
        if 'coverage_speed' in feature_set_features_list:
            self.compute_single_bots_coverage_speed()

    def compute_cluster_trajectory(self) -> np.ndarray:
        all_bot_trajectory = np.asarray([bot.single_robot_positions for bot in self.list_of_footbots])
        return np.mean(all_bot_trajectory, axis=0)

    def compute_cluster_speed(self):
        self.speed_time_series = FootBot.compute_entity_speed(self.timesteps, self.trajectory)

    def compute_distances_from_centroid(self):
        for bot in self.list_of_footbots:
            bot.compute_distance_from_centroid(self.trajectory)
            bot.compute_cumulative_distance_from_centroid()

    def compute_area_coverage(self):

        area_coverage = {}

        for area_subdivision in self.area_partitions:
            positions_index = 0

            percentage_time_series = []

            while any(not partition.visited for partition in area_subdivision) \
                    and positions_index < len(self.trajectory):

                for bot in self.list_of_footbots:
                    not_visited_partitions = [partition for partition in area_subdivision if not partition.visited]

                    for partition in not_visited_partitions:
                        position = bot.single_robot_positions[positions_index]
                        if (partition.left_bound <= position[0] < partition.right_bound and
                                partition.low_bound <= position[1] < partition.top_bound):
                            partition.visited = True

                percentage_time_series.append(
                    len([partition for partition in area_subdivision if partition.visited]) / len(area_subdivision)
                )

                positions_index += 1

            if positions_index < len(self.trajectory):
                percentage_time_series.extend([percentage_time_series[-1]] * (len(self.trajectory) - positions_index))

            area_coverage[str(len(area_subdivision))] = np.asarray(percentage_time_series)

        return area_coverage

    def compute_area_partitions(self):
        left_bound = min([np.min(bot.single_robot_positions[..., 0]) for bot in self.list_of_footbots])
        right_bound = max([np.max(bot.single_robot_positions[..., 0]) for bot in self.list_of_footbots])
        low_bound = min([np.min(bot.single_robot_positions[..., 1]) for bot in self.list_of_footbots])
        top_bound = max([np.max(bot.single_robot_positions[..., 1]) for bot in self.list_of_footbots])

        area_partitions = []

        for split_number in Parser.read_area_splits():
            horizontal_splits = [left_bound + abs(right_bound - left_bound) / split_number * repetitions
                                 for repetitions in range(split_number + 1)]
            vertical_splits = [low_bound + abs(top_bound - low_bound) / split_number * repetitions for repetitions in
                               range(split_number + 1)]

            area_partitions.append([AreaPartition(left_bound=horizontal_splits[i],
                                                  right_bound=horizontal_splits[i + 1],
                                                  low_bound=vertical_splits[j],
                                                  top_bound=vertical_splits[j + 1])
                                    for i in range(len(horizontal_splits) - 1)
                                    for j in range(len(vertical_splits) - 1)
                                    ])

        return area_partitions

    def compute_single_bots_area_coverage(self):
        for bot in self.list_of_footbots:
            print('Computing Coverage Speed bot:' + str(bot.identifier))
            bot.compute_area_coverage(self.area_partitions)

    def compute_single_bots_coverage_speed(self):
        for bot in self.list_of_footbots:
            bot.compute_coverage_speed()
