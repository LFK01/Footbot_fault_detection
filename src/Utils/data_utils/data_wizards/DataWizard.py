import numpy as np
from src.Utils.Parser import Parser
from src.classes.Swarm import Swarm
from src.classes.FootBot import FootBot
from src.Utils.data_utils.datasets.TrValTeDataset import TrValTeDataset
from src.Utils.data_utils.datasets.GeneralDataset import GeneralDataset


class DataWizard:
    def __init__(self, timesteps: int,
                 time_window: int,
                 label_size: int,
                 experiments: list[Swarm],
                 down_sampling_steps: int = 1,
                 preprocessing_type: str = 'raw'):
        self.timesteps: int = timesteps - 1
        self.time_window: int = time_window
        self.label_size: int = label_size
        self.experiments: list[Swarm] = experiments
        self.down_sampling_steps = down_sampling_steps
        self.preprocessing_type = preprocessing_type

    @staticmethod
    def shortest_experiment_timesteps(experiment_list: list[Swarm]) -> int:
        return min(
            experiment.list_of_footbots[0].number_of_timesteps for experiment in experiment_list
        )

    @staticmethod
    def slice_train_test_experiments(bot: int,
                                     train_experiments,
                                     test_experiments,
                                     down_sampling_steps: int = 1) -> GeneralDataset:

        """
        VALIDATION SET NOT COMPUTED
        Method that retrieves the features of each bot and stacks them in order to be analyzed by the learning model.
        Experiments are concatenated one after the other in order to create an unique time series.
        Parameters
        ----------
        bot: int = the current bot to analyze
        train_experiments
        test_experiments
        down_sampling_steps

        Returns
        -------

        """

        bot_train_dataset, bot_train_target_dataset = DataWizard.retrieve_features_and_target(
            bot=bot,
            experiments=train_experiments,
            down_sampling_steps=down_sampling_steps
        )
        bot_test_dataset, bot_test_target_dataset = DataWizard.retrieve_features_and_target(
            bot=bot,
            experiments=test_experiments,
            down_sampling_steps=down_sampling_steps
        )
        # numpy concatenate creates an unique time series of all the experiments
        bot_dataset = GeneralDataset(
            bot_identifier=bot,
            train_dataset=np.concatenate(bot_train_dataset, axis=-1),
            target_train_dataset=np.concatenate(bot_train_target_dataset, axis=-1),
            test_dataset=np.concatenate(bot_test_dataset, axis=-1),
            target_test_dataset=np.concatenate(bot_test_target_dataset, axis=-1)
        )
        return bot_dataset

    @staticmethod
    def slice_train_val_test_experiments(bot: int,
                                         train_experiments,
                                         validation_experiments,
                                         test_experiments,
                                         down_sampling_steps: int = 1) -> TrValTeDataset:

        """
        VALIDATION SET COMPUTED
        Method that retrieves the features of each bot and stacks them in order to be analyzed by the learning model.
        Parameters
        ----------
        bot: int = the current bot to analyze
        train_experiments
        validation_experiments
        test_experiments
        down_sampling_steps

        Returns
        -------

        """

        bot_train_dataset, bot_train_target_dataset = DataWizard.retrieve_features_and_target(
            bot=bot,
            experiments=train_experiments,
            down_sampling_steps=down_sampling_steps
        )
        bot_val_dataset, bot_val_target_dataset = DataWizard.retrieve_features_and_target(
            bot=bot,
            experiments=validation_experiments,
            down_sampling_steps=down_sampling_steps
        )
        bot_test_dataset, bot_test_target_dataset = DataWizard.retrieve_features_and_target(
            bot=bot,
            experiments=test_experiments,
            down_sampling_steps=down_sampling_steps
        )

        bot_dataset = TrValTeDataset(
            bot_identifier=bot,
            train_dataset=np.concatenate(bot_train_dataset, axis=-1),
            target_train_dataset=np.concatenate(bot_train_target_dataset, axis=-1),
            validation_dataset=np.concatenate(bot_val_dataset, axis=-1),
            target_validation_dataset=np.concatenate(bot_val_target_dataset, axis=-1),
            test_dataset=np.concatenate(bot_test_dataset, axis=-1),
            target_test_dataset=np.concatenate(bot_test_target_dataset, axis=-1)
        )
        return bot_dataset

    @staticmethod
    def retrieve_bot_features(bot: FootBot,
                              swarm: Swarm,
                              down_sampling_steps: int) -> list[np.ndarray]:

        features_list = Parser.read_features_set()

        vector = []
        if 'single_robot_positions' in features_list:
            vector.append(bot.single_robot_positions[::down_sampling_steps, 0])
            vector.append(bot.single_robot_positions[::down_sampling_steps, 1])
        if 'speed_time_series' in features_list:
            vector.append(bot.speed_time_series[::down_sampling_steps])
        if 'direction_time_series' in features_list:
            vector.append(bot.direction_time_series[::down_sampling_steps, 0])
            vector.append(bot.direction_time_series[::down_sampling_steps, 1])
        if 'cumulative_speed' in features_list:
            vector.append(bot.cumulative_speed[::down_sampling_steps])
        if 'neighbors_time_series' in features_list:
            vector.append(bot.neighbors_time_series[::down_sampling_steps])
        if 'swarm_cohesion_time_series' in features_list:
            vector.append(bot.swarm_cohesion_time_series[::down_sampling_steps])
        if 'distance_from_centroid_time_series' in features_list:
            vector.append(bot.distance_from_centroid_time_series[::down_sampling_steps])
        if 'cumulative_distance_from_centroid_time_series' in features_list:
            vector.append(bot.cumulative_distance_from_centroid_time_series[::down_sampling_steps])
        if 'positions_entropy' in features_list:
            vector.append(bot.positions_entropy[::down_sampling_steps])
        if 'area_coverage' in features_list:
            for area_coverage_slice in bot.area_coverage:
                vector.append(area_coverage_slice[::down_sampling_steps])
        if 'coverage_speed' in features_list:
            for coverage_speed_slice in bot.coverage_speed:
                vector.append(coverage_speed_slice[::down_sampling_steps])
        if 'global_features' in features_list:
            vector.append(swarm.trajectory[::down_sampling_steps, 0])
            vector.append(swarm.trajectory[::down_sampling_steps, 1])
            vector.append(swarm.speed_time_series[::down_sampling_steps])
            for area_coverage_slice in swarm.area_coverage.values():
                vector.append(area_coverage_slice[::down_sampling_steps])

        # since some arrays may end up having different lengths, we short them to the maximum common length
        # which is the minimum among the length of all the features
        cut_max_common_length = min(feature.shape[0] for feature in vector)
        vector = [feature[:cut_max_common_length] for feature in vector]

        return vector

    @staticmethod
    def retrieve_features_and_target(bot: int, experiments: list[Swarm], down_sampling_steps: int = 1):
        bot_dataset = []
        bot_target_dataset = []

        for exp in experiments:
            retrieved_features = np.asarray(DataWizard.retrieve_bot_features(bot=exp.list_of_footbots[bot],
                                                                             swarm=exp,
                                                                             down_sampling_steps=down_sampling_steps))

            bot_dataset.append(retrieved_features)
            bot_target_dataset.append(exp.list_of_footbots[bot].fault_time_series[::down_sampling_steps])

        return bot_dataset, bot_target_dataset
