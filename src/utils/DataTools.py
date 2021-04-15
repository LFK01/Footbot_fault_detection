import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.Classes.Swarm import Swarm


class DataWizard:
    def __init__(self):
        pass

    @staticmethod
    def prepare_raw_input(swarm: Swarm):
        return DataWizard.build_data_vector(swarm=swarm)

    @staticmethod
    def prepare_normalized_data(swarm: Swarm):
        vector = DataWizard.build_data_vector(swarm=swarm)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        return scaler.fit_transform(vector)

    @staticmethod
    def prepare_standardized_data(swarm: Swarm):
        data_vector = DataWizard.build_data_vector(swarm=swarm)
        scaler = StandardScaler()
        return scaler.fit_transform(X=data_vector)

    @staticmethod
    def build_data_vector(swarm: Swarm):
        data_vector = []
        # append all robots features
        for bot in swarm.list_of_footbots:
            data_vector.append(bot.single_robot_positions[:, 0])
            data_vector.append(bot.single_robot_positions[:, 1])
            data_vector.append(bot.traversed_distance_time_series)
            data_vector.append(bot.direction_time_series[:, 0])
            data_vector.append(bot.direction_time_series[:, 1])
            data_vector.append(bot.cumulative_traversed_distance)
            data_vector.append(bot.neighbors_time_series)
            data_vector.append(bot.swarm_cohesion_time_series)
            data_vector.append(bot.distance_from_centroid_time_series)

        data_vector.append(swarm.trajectory[:, 0])
        data_vector.append(swarm.trajectory[:, 1])
        data_vector.append(swarm.traversed_distance_time_series)

        return data_vector

    @staticmethod
    def prepare_target(swarm: Swarm):
        target = []
        for bot in swarm.list_of_footbots:
            target.append(bot.fault_time_series)
        return np.asarray(target)
