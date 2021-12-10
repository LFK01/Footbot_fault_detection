import os
import pickle
from os.path import join, exists
from os import makedirs, listdir
from datetime import datetime
from PIL import Image
from typing import List, Tuple

import matplotlib.pyplot as plt
from src.classes.FootBot import FootBot
from src.classes.Swarm import Swarm
from src.Utils.Parser import Parser


class Plotter:
    """
    Class used to plot data
    """

    def __init__(self):
        """
        Empty constructor
        """
        pass

    @staticmethod
    def plot_trajectories(footbot_list: List[FootBot],
                          swarm: Swarm,
                          plot_swarm: bool,
                          min_x: float,
                          min_y: float,
                          max_x: float,
                          max_y: float,
                          path: str = "",
                          title: str = "Trajectory of each bot and centroid of cluster",
                          show_plot: bool = True) -> None:
        """
        Method to plot trajectories of each robot. Shows the plot.

        Parameters
        ----------
        swarm: Swarm
            Swarm object to retrieve cluster features
        footbot_list : list
            List of FootBot instances
        plot_swarm: bool
            Boolean variable to decide if you want to plot the trajectory of the swarm centroid
        path: str
            Additional string to specify where to save the plot
        title: str
            Title that shows on the pyplot graph
        show_plot: bool
            Boolean parameter to decide if the plot has to be shown during execution or only saved
        """

        fig = plt.figure(figsize=(8, 8), dpi=80)
        for bot in footbot_list:
            plt.scatter(
                [pos[0] for pos in bot.single_robot_positions],
                [pos[1] for pos in bot.single_robot_positions],
                s=0.1, label=bot.identifier)
        if plot_swarm:
            plt.scatter(
                [pos[0] for pos in swarm.trajectory],
                [pos[1] for pos in swarm.trajectory],
                facecolors='none', edgecolors='r', s=0.2, label='swarm'
            )
        plt.xlim([min_x-1, max_x+1])
        plt.ylim([min_y-1, max_y+1])
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(title)
        # plt.legend(loc="upper right", markerscale=18)
        path = join(path, title.replace(" ", "_"))
        plt.savefig(path)
        if show_plot:
            plt.show()
        plt.close(fig)

    @staticmethod
    def plot_trajectory_entropy(footbot_list: List[FootBot],
                                path: str = "",
                                title: str = 'Entropy for each bot',
                                show_plot: bool = True) -> None:
        """
        Method to plot the entropy of the trajectory. Shows the plot.

        Parameters
        ----------
        footbot_list : list
            List of FootBot instances
        path: str
            Additional string to specify where to save the plot
        title: str
            Title that shows on the pyplot graph
        show_plot: bool
            Boolean parameter to decide if the plot has to be shown during execution or only saved
        """
        fig = plt.figure()
        for bot in footbot_list[::4]:
            plt.plot(bot.positions_entropy[:3000], alpha=0.5)
        plt.xlabel("Timestep")
        plt.ylabel("Entropy")
        plt.title(title)
        if path != "":
            path += "/"
        plt.savefig(path + title.replace(" ", "_"))
        if show_plot:
            plt.show()
        plt.close(fig)

    @staticmethod
    def plot_state_time_series(footbot_list: List[FootBot],
                               path: str = "",
                               title: str = "State for each bot",
                               show_plot: bool = True) -> None:
        """
        Method to plot state of each robot. Shows the plot.

        Parameters
        ----------
        footbot_list : list
            List of FootBot instances
        path: str
            Additional string to specify where to save the plot
        title: str
            Title of the graph
        show_plot: bool
            Boolean parameter to decide if the plot has to be shown during execution or only saved
        """
        fig = plt.figure()
        for bot in footbot_list:
            plt.plot(bot.state_time_series)
        plt.xlabel("Timestep")
        plt.ylabel("State")
        plt.title(title)
        plt.text(0, 0, 'STATE = Resting')
        plt.text(0, 1, 'STATE = Exploring')
        plt.text(0, 2, 'STATE = Returning to nest')
        if path != "":
            path += "/"
        plt.savefig(path + title.replace(" ", "_"))
        if show_plot:
            plt.show()
        plt.close(fig)

    @staticmethod
    def plot_time_rested_time_series(footbot_list: List[FootBot],
                                     path: str = "",
                                     title: str = "Rested Time for each bot",
                                     show_plot: bool = True) -> None:
        """
        Method to plot rested_time of each robot. Shows the plot.

        Parameters
        ----------
        footbot_list : list
            List of FootBot instances
        path: str
            Additional string to specify where to save the plot
        title: str
            Title of the graph
        show_plot: bool
            Boolean parameter to decide if the plot has to be shown during execution or only saved
        """
        fig = plt.figure()
        for bot in footbot_list:
            plt.plot(bot.TimeRested_time_series)
        plt.xlabel("Timestep")
        plt.ylabel("Rested Time")
        plt.title(title)
        if path != "":
            path += "/"
        plt.savefig(path + title.replace(" ", "_"))
        if show_plot:
            plt.show()
        plt.close(fig)

    @staticmethod
    def plot_has_food_time_series(footbot_list: List[FootBot],
                                  path: str = "",
                                  title: str = "Has food info for each bot",
                                  show_plot: bool = True) -> None:
        """
        Method to plot if each robot has food. Shows the plot.

        Parameters
        ----------
        footbot_list : list
            List of FootBot instances
        path: str
            Additional string to specify where to save the plot
        title: str
            Title of the graph
        show_plot: bool
            Boolean parameter to decide if the plot has to be shown during execution or only saved
        """
        fig = plt.figure()
        for bot in footbot_list:
            plt.plot(bot.HasFood_time_series)
        plt.xlabel("Timestep")
        plt.ylabel("Does it have food?")
        plt.title(title)
        if path != "":
            path += "/"
        plt.savefig(path + title.replace(" ", "_"))
        if show_plot:
            plt.show()
        plt.close(fig)

    @staticmethod
    def plot_total_food_time_series(footbot_list: List[FootBot],
                                    path: str = "",
                                    title: str = "Collected food for each bot",
                                    show_plot: bool = True) -> None:
        """
        Method to plot total food collected from each bot. Shows the plot.

        Parameters
        ----------
        footbot_list : list
            List of FootBot instances
        path: str
            Additional string to specify where to save the plot
        title: str
            Title of the graph
        show_plot: bool
            Boolean parameter to decide if the plot has to be shown during execution or only saved
        """
        fig = plt.figure()
        for bot in footbot_list:
            plt.plot(bot.TotalFood_time_series)
        plt.xlabel("Timestep")
        plt.ylabel("Collected food")
        plt.title(title)
        if path != "":
            path += "/"
        plt.savefig(path + title.replace(" ", "_"))
        if show_plot:
            plt.show()
        plt.close(fig)

    @staticmethod
    def plot_time_searching_for_nest_time_series(footbot_list: List[FootBot],
                                                 path: str = "",
                                                 title: str = "Time to search space in nest",
                                                 show_plot: bool = True) -> None:
        """
        Method to plot if each robot has food. Shows the plot.

        Parameters
        ----------
        footbot_list : list
            List of FootBot instances
        path: str
            Additional string to specify where to save the plot
        title: str
            Title of the graph
        show_plot: bool
            Boolean parameter to decide if the plot has to be shown during execution or only saved
        """
        fig = plt.figure()
        for bot in footbot_list:
            plt.plot(bot.TimeSearchingForNest_time_series)
        plt.xlabel("Timestep")
        plt.ylabel("Nest Time")
        plt.title(title)
        if path != "":
            path += "/"
        plt.savefig(path + title.replace(" ", "_"))
        if show_plot:
            plt.show()
        plt.close(fig)

    @staticmethod
    def plot_exploring_unsuccessfully_time_series(footbot_list: List[FootBot],
                                                  path: str = "",
                                                  title: str = "Time to search space in nest",
                                                  show_plot: bool = True) -> None:
        """
        Method to plot if each robot has food. Shows the plot.

        Parameters
        ----------
        footbot_list : list
            List of FootBot instances
        path: str
            Additional string to specify where to save the plot
        title: str
            Title of the graph
        show_plot: bool
            Boolean parameter to decide if the plot has to be shown during execution or only saved
        """
        fig = plt.figure()
        for bot in footbot_list:
            plt.plot(bot.TimeExploringUnsuccessfully_time_series)
        plt.xlabel("Timestep")
        plt.ylabel("Exploration time")
        plt.title(title)
        if path != "":
            path += "/"
        plt.savefig(path + title.replace(" ", "_"))
        if show_plot:
            plt.show()
        plt.close(fig)

    @staticmethod
    def plot_speeds(footbot_list: List[FootBot],
                    path: str = "",
                    title: str = 'Instantaneous Traversed Distance for each bot',
                    show_plot: bool = True) -> None:
        """
        Method to plot speeds of each robot. Shows the plot.

        Parameters
        ----------
        footbot_list : list
            List of FootBot instances
        path: str
            Additional string to specify where to save the plot
        title: str
            Title that shows on the pyplot graph
        show_plot: bool
            Boolean parameter to decide if the plot has to be shown during execution or only saved
        """
        fig = plt.figure()
        for bot in footbot_list:
            plt.plot(bot.speed_time_series)
        plt.xlabel("Timestep")
        plt.ylabel("Speed")
        plt.title(title)
        if path != "":
            path += "/"
        plt.savefig(path + title.replace(" ", "_"))
        if show_plot:
            plt.show()
        plt.close(fig)

    @staticmethod
    def plot_cumulative_traversed_distance(footbot_list: List[FootBot],
                                           path: str = "",
                                           title: str = 'Cumulative Traversed Distance for each bot',
                                           show_plot: bool = True) -> None:
        """
        Method to plot cumulative of each robot. Shows the plot.

        Parameters
        ----------
        footbot_list : list
            List of FootBot instances
        path: str
            Additional string to specify where to save the plot
        title: str
            Title that shows on the pyplot graph
        show_plot: bool
            Boolean parameter to decide if the plot has to be shown during execution or only saved
        """
        fig = plt.figure()
        for bot in footbot_list:
            plt.plot(bot.cumulative_speed)
        plt.xlabel("Timestep")
        plt.ylabel("Cumulative Traversed Distance")
        plt.title(title)
        if path != "":
            path += "/"
        plt.savefig(path + title.replace(" ", "_"))
        if show_plot:
            plt.show()
        plt.close(fig)

    @staticmethod
    def plot_neighbors(footbot_list: List[FootBot],
                       path: str = "",
                       title: str = 'Number of Neighbors for each bot',
                       show_plot: bool = True) -> None:
        """
        Method to plot neighbors of each robot. Shows the plot.

        Parameters
        ----------
        footbot_list : list
            List of FootBot instances
        path: str
            Additional string to specify where to save the plot
        title: str
            Title that shows on the pyplot graph
        show_plot: bool
            Boolean parameter to decide if the plot has to be shown during execution or only saved
        """
        fig = plt.figure()
        for bot in footbot_list:
            plt.plot(bot.neighbors_time_series)
        plt.xlabel("Timestep")
        plt.ylabel("Number of Neighbors")
        plt.title(title)
        if path != "":
            path += "/"
        plt.savefig(path + title.replace(" ", "_"))
        if show_plot:
            plt.show()
        plt.close(fig)

    @staticmethod
    def plot_faulty_robots(footbot_list: List[FootBot],
                           path: str = "",
                           title: str = "Number of faulty bots",
                           show_plot: bool = True) -> None:
        """
        Method to plot neighbors of each robot. Shows the plot.

        Parameters
        ----------
        footbot_list : list
            List of FootBot instances
        path: str
            Additional string to specify where to save the plot
        title: str
            Title that shows on the pyplot graph
        show_plot: bool
            Boolean parameter to decide if the plot has to be shown during execution or only saved
        """
        cumulative_faulty_bots = []
        fig = plt.figure()
        for timestep in range(len(footbot_list[0].fault_time_series)):
            cumulative_faulty_bots.append(
                sum(bot.fault_time_series[timestep] for bot in footbot_list)
            )
        plt.plot(cumulative_faulty_bots)
        plt.xlabel("Timestep")
        plt.ylabel("Number of faulty bots")
        plt.title(title)
        if path != "":
            path += "/"
        plt.savefig(path + title.replace(" ", "_"))
        if show_plot:
            plt.show()
        plt.close(fig)

    @staticmethod
    def plot_swarm_cohesion(footbot_list: List[FootBot],
                            path: str = "",
                            title: str = "Average distance from all bots for each bot",
                            show_plot: bool = True) -> None:
        """
        Method to plot cumulative of each robot. Shows the plot.

        Parameters
        ----------
        footbot_list : List[FootBot]
            List of FootBot instances
        path: str
            Additional string to specify where to save the plot
        title: str
            Title that shows on the pyplot graph
        show_plot: bool
            Boolean parameter to decide if the plot has to be shown during execution or only saved
        """
        fig = plt.figure()
        for bot in footbot_list:
            plt.plot(bot.swarm_cohesion_time_series)
        plt.xlabel("Timestep")
        plt.ylabel("Average Distance")
        plt.title(title)
        if path != "":
            path += "/"
        plt.savefig(path + title.replace(" ", "_"))
        if show_plot:
            plt.show()
        plt.close(fig)

    @staticmethod
    def plot_swarm_speed(footbot_list: Swarm,
                         path: str = "",
                         title: str = "Swarm Speed",
                         show_plot: bool = True):
        """
        Method to plot trajectory of swarm. Shows the plot.

        Parameters
        ----------
        footbot_list : Swarm
            instance of the swarm of bots
        path: str
            Additional string to specify where to save the plot
        title: str
            Title that shows on the pyplot graph
        show_plot: bool
            Boolean parameter to decide if the plot has to be shown during execution or only saved
        """
        fig = plt.figure()
        plt.plot(footbot_list.speed_time_series)
        plt.xlabel("timestep")
        plt.ylabel("Swarm speed")
        plt.title(title)
        if path != "":
            path += "/"
        plt.savefig(path + title.replace(" ", "_"))
        if show_plot:
            plt.show()
        plt.close(fig)

    @staticmethod
    def plot_distances_from_centroid(footbot_list: List[FootBot],
                                     path: str = "",
                                     additional_title_string: str = "",
                                     show_plot: bool = True) -> None:
        """
        Method to plot distances from centroid. Shows the plot.

        Parameters
        ----------
        footbot_list : list
            List of FootBot instances
        path: str
            Additional string to specify where to save the plot
        additional_title_string: str
            Title that shows on the pyplot graph
        show_plot: bool
            Boolean parameter to decide if the plot has to be shown during execution or only saved
        """
        fig = plt.figure()
        for bot in footbot_list:
            plt.plot(bot.distance_from_centroid_time_series)
        plt.xlabel("Timestep")
        plt.ylabel("Distance from Centroid")
        title = additional_title_string + " Distance from centroid for each bot"
        plt.title(title)
        if path != "":
            path += "/"
        plt.savefig(path + title.replace(" ", "_"))
        if show_plot:
            plt.show()
        plt.close(fig)

        fig = plt.figure()
        for bot in footbot_list:
            plt.plot(bot.cumulative_distance_from_centroid_time_series)
        plt.xlabel("Timestep")
        plt.ylabel("Cumulative Distance from Centroid")
        title = additional_title_string + " Cumulative Distance from centroid"
        plt.title(title)
        if path != "":
            path += "/"
        plt.savefig(path + title.replace(" ", "_"))
        if show_plot:
            plt.show()
        plt.close(fig)

    @staticmethod
    def plot_footbot_area_coverage(footbot_list: List[FootBot],
                                   path: str = "",
                                   additional_title_string: str = "",
                                   show_plot: bool = True) -> None:
        """
        Method to plot distances from centroid. Shows the plot.

        Parameters
        ----------
        footbot_list : list
            List of FootBot instances
        path: str
            Additional string to specify where to save the plot
        additional_title_string: str
            Additional string to specify whether the bot is nominal or not
        show_plot: bool
            Boolean parameter to decide if the plot has to be shown during execution or only saved
        """
        area_splits = Parser.read_area_splits()
        for percentage_index in range(len(area_splits)):
            fig = plt.figure()
            for bot in footbot_list:
                plt.plot(bot.area_coverage[percentage_index])
            plt.xlabel("Timestep")
            plt.ylabel("Coverage Percentage")
            plt.ylim((-0.1, 1.1))
            title = additional_title_string + " Single Bot Area Coverage with " \
                    + str(area_splits[percentage_index] ** 2) \
                    + " subdivisions"
            plt.title(title)
            if path != "":
                path += "/"
            plt.savefig(path + title.strip().replace(" ", "_"))
            if show_plot:
                plt.show()
            plt.close(fig)

        for percentage_index in range(len(area_splits)):
            fig = plt.figure()
            for bot in footbot_list:
                plt.plot(bot.coverage_speed[percentage_index])
            plt.xlabel("Timestep")
            plt.ylabel("Coverage Speed")
            title = additional_title_string + " Single Bot Area Coverage Speed with " \
                    + str(area_splits[percentage_index] ** 2) \
                    + " subdivisions"
            plt.title(title)
            if path != "":
                path += "/"
            plt.savefig(path + title.strip().replace(" ", "_"))
            if show_plot:
                plt.show()
            plt.close(fig)

    @staticmethod
    def plot_swarm_area_coverage(main_swarm: Swarm,
                                 path: str = "",
                                 additional_title_string: str = "",
                                 show_plot: bool = True) -> None:
        """
        Method to plot distances from centroid. Shows the plot.

        Parameters
        ----------
        main_swarm: Swarm
            Swarm of all the bots to plot the cumulative area coverage
        path: str
            Additional string to specify where to save the plot
        additional_title_string: str
            Additional string to specify whether the bot is nominal or not
        show_plot: bool
            Boolean parameter to decide if the plot has to be shown during execution or only saved
        """
        for key_area_coverage in main_swarm.area_coverage.keys():
            fig = plt.figure()
            plt.plot(main_swarm.area_coverage[key_area_coverage])
            plt.xlabel("Timestep")
            plt.ylabel("Coverage Percentage")
            plt.ylim((-0.1, 1.1))
            title = additional_title_string + " Swarm Area Coverage with " + str(key_area_coverage) + " subdivisions"
            plt.title(title)
            if path != "":
                path += "/"
            plt.savefig(path + title.strip().replace(" ", "_"))
            if show_plot:
                plt.show()
            plt.close(fig)

    @staticmethod
    def plot_speed_neighbors_position_entr(main_swarm: Swarm,
                                           saving_path: str,
                                           bot_index: int,
                                           show_plot: bool = True) -> None:
        """
        Method to plot 3 timeseries together. Shows the plot.

        Parameters
        ----------
        bot_index: int
            index of the bot to plot its timeseries
        main_swarm: Swarm
            Swarm of all the bots to plot the cumulative area coverage
        saving_path: str
            Additional string to specify where to save the plot
        show_plot: bool
            Boolean parameter to decide if the plot has to be shown during execution or only saved
        """
        title = 'Bot {} multivariate timeseries'.format(bot_index)
        fig, (ax1, ax2, ax3) = plt.subplots(3)
        fig.suptitle(title)
        ax1.plot(main_swarm.list_of_footbots[bot_index].speed_time_series)
        ax1.set_ylabel('Speed')
        ax2.plot(main_swarm.list_of_footbots[bot_index].neighbors_time_series)
        ax2.set_ylabel('Neighbors')
        ax3.plot(main_swarm.list_of_footbots[bot_index].positions_entropy)
        ax3.set_xlabel('Timestep')
        ax3.set_ylabel('Entropy')

        plt.savefig(join(saving_path, title.strip().replace(" ", "_") + '_bot{}'.format(bot_index)))
        if show_plot:
            plt.show()
        plt.close(fig)

    @staticmethod
    def make_folder_from_json(par_task_name: str, file_number: int) -> str:
        new_folder_name = Parser.read_filename(task_name=par_task_name,
                                               file_number=file_number).split('/')[-1].split('.')[0]
        # find project root
        root = Parser.get_project_root()
        saving_graphs_file_path = join(root, 'images', new_folder_name)

        if exists(saving_graphs_file_path):
            saving_graphs_file_path = saving_graphs_file_path + '_' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        makedirs(saving_graphs_file_path)

        return saving_graphs_file_path

    @staticmethod
    def make_folder_from_complete_file_path(saving_graphs_file_path: str) -> str:
        if exists(saving_graphs_file_path):
            saving_graphs_file_path = saving_graphs_file_path + '_' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        makedirs(saving_graphs_file_path)

        return saving_graphs_file_path

    @staticmethod
    def divide_flocks(footbots_list: List[FootBot]) -> Tuple[List[FootBot], List[FootBot]]:
        faulty_bots = [bot for bot in footbots_list if any(bot.fault_time_series)]
        nominal_bots = [bot for bot in footbots_list if not any(bot.fault_time_series)]

        return faulty_bots, nominal_bots

    @staticmethod
    def build_generic_swarm(par_task_name: str,
                            feature_set_features_list: List[str],
                            file_number: int):
        neighborhood_radius = Parser.read_neighborhood_radius()
        time_window_size = Parser.read_time_window()
        file = Parser.read_filename(task_name=par_task_name, file_number=file_number)
        timesteps = Parser.retrieve_timesteps_series_from_dataframe(
            df_footbot_positions=Parser.open_pandas_dataframe(filename=file,
                                                              task_name=par_task_name)
        )
        footbots_list = Parser.create_generic_swarm(task_name=par_task_name,
                                                    feature_set_features_list=feature_set_features_list,
                                                    filename=file,
                                                    neighborhood_radius=neighborhood_radius,
                                                    time_window_size=time_window_size)
        return footbots_list, Swarm(timesteps=timesteps,
                                    feature_set_features_list=feature_set_features_list,
                                    swarm=footbots_list)

    @staticmethod
    def build_foraging_swarm(par_task_name: str,
                             feature_set_features_list: List[str],
                             file_number: int):
        neighborhood_radius = Parser.read_neighborhood_radius()
        time_window_size = Parser.read_time_window()
        file = Parser.read_filename(task_name=par_task_name, file_number=file_number)
        timesteps = Parser.retrieve_timesteps_series_from_dataframe(
            df_footbot_positions=Parser.open_pandas_dataframe(filename=file,
                                                              task_name=par_task_name)
        )

        footbots_list = Parser.create_foraging_swarm(filename=file,
                                                     feature_set_features_list=feature_set_features_list,
                                                     neighborhood_radius=neighborhood_radius,
                                                     time_window_size=time_window_size)
        root = Parser.get_project_root()
        path = join(root, 'cached_files')
        with open(join(path, 'foraging_footbot_list' +
                             datetime.now().strftime('%d-%m-%Y_%H-%M') +
                             '70_bots_800sec.pkl'),
                  'wb') as output_file:
            pickle.dump(footbots_list, output_file)

        return footbots_list, Swarm(timesteps=timesteps,
                                    feature_set_features_list=feature_set_features_list,
                                    swarm=footbots_list)

    @staticmethod
    def plot_traj_and_fault_in_separate_window(nominal_bots: List[FootBot],
                                               faulty_bots: List[FootBot],
                                               saving_path: str,
                                               main_swarm: Swarm,
                                               show_graphs: bool,
                                               title: str = None):
        min_x = min([min(bot.single_robot_positions[0]) for bot in main_swarm.list_of_footbots])
        min_y = min([min(bot.single_robot_positions[1]) for bot in main_swarm.list_of_footbots])
        max_x = max([max(bot.single_robot_positions[0]) for bot in main_swarm.list_of_footbots])
        max_y = max([max(bot.single_robot_positions[1]) for bot in main_swarm.list_of_footbots])
        # min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y,
        print('Plotting trajectories')
        if title is not None:
            Plotter.plot_trajectories(footbot_list=nominal_bots,
                                      swarm=main_swarm,
                                      plot_swarm=True,
                                      path=saving_path,
                                      min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y,
                                      show_plot=show_graphs,
                                      title='nom' + title)
            Plotter.plot_trajectories(footbot_list=faulty_bots,
                                      swarm=main_swarm,
                                      plot_swarm=False,
                                      path=saving_path,
                                      min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y,
                                      show_plot=show_graphs,
                                      title='fault' + title)
            Plotter.plot_faulty_robots(footbot_list=main_swarm.list_of_footbots,
                                       path=saving_path,
                                       show_plot=show_graphs,
                                       title='num' + title)
        else:
            Plotter.plot_trajectories(footbot_list=nominal_bots,
                                      swarm=main_swarm,
                                      plot_swarm=True,
                                      path=saving_path,
                                      min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y,
                                      show_plot=False)
            Plotter.plot_trajectories(footbot_list=faulty_bots,
                                      swarm=main_swarm,
                                      plot_swarm=False,
                                      path=saving_path,
                                      min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y,
                                      show_plot=False)
            Plotter.plot_faulty_robots(footbot_list=main_swarm.list_of_footbots,
                                       path=saving_path,
                                       show_plot=False)

    @staticmethod
    def plot_uinv_multiv_ts(saving_path: str,
                            main_swarm: Swarm,
                            show_graphs: bool):
        print('Plotting univ multiv ts')
        for bot in range(len(main_swarm.list_of_footbots)):
            Plotter.plot_speeds(footbot_list=[main_swarm.list_of_footbots[bot]],
                                path=saving_path,
                                title='Bot {} speed'.format(bot),
                                show_plot=show_graphs)
            Plotter.plot_speed_neighbors_position_entr(main_swarm=main_swarm,
                                                       bot_index=bot,
                                                       saving_path=saving_path,
                                                       show_plot=show_graphs)

    @staticmethod
    def plot_common_features(nominal_bots: List[FootBot],
                             faulty_bots: List[FootBot],
                             main_swarm: Swarm,
                             saving_graphs_file_path,
                             show_all_graphs: bool = True):
        min_x = min([min(bot.single_robot_positions[..., 0]) for bot in main_swarm.list_of_footbots])
        min_y = min([min(bot.single_robot_positions[..., 1]) for bot in main_swarm.list_of_footbots])
        max_x = max([max(bot.single_robot_positions[..., 0]) for bot in main_swarm.list_of_footbots])
        max_y = max([max(bot.single_robot_positions[..., 1]) for bot in main_swarm.list_of_footbots])
        Plotter.plot_trajectories(footbot_list=nominal_bots,
                                  swarm=main_swarm,
                                  plot_swarm=True,
                                  min_x=min_x,
                                  min_y=min_y,
                                  max_x=max_x,
                                  max_y=max_y,
                                  path=saving_graphs_file_path,
                                  title="Nominal Bots Trajectories",
                                  show_plot=show_all_graphs)
        Plotter.plot_trajectories(footbot_list=faulty_bots,
                                  swarm=main_swarm,
                                  plot_swarm=True,
                                  min_x=min_x,
                                  min_y=min_y,
                                  max_x=max_x,
                                  max_y=max_y,
                                  path=saving_graphs_file_path,
                                  title="Fault Bots Trajectories",
                                  show_plot=show_all_graphs)
        Plotter.plot_speeds(footbot_list=nominal_bots,
                            path=saving_graphs_file_path,
                            title="Nominal Bots Speeds",
                            show_plot=show_all_graphs)
        Plotter.plot_speeds(footbot_list=faulty_bots,
                            path=saving_graphs_file_path,
                            title="Fault Bots Speeds",
                            show_plot=show_all_graphs)
        Plotter.plot_cumulative_traversed_distance(footbot_list=nominal_bots,
                                                   path=saving_graphs_file_path,
                                                   title="Nominal Bots Cumulative Speeds",
                                                   show_plot=show_all_graphs)
        Plotter.plot_cumulative_traversed_distance(footbot_list=faulty_bots,
                                                   path=saving_graphs_file_path,
                                                   title="Fault Bots Cumulative Speeds",
                                                   show_plot=show_all_graphs)
        Plotter.plot_trajectory_entropy(footbot_list=nominal_bots,
                                        path=saving_graphs_file_path,
                                        title="Nominal Bots Trajectory Entropy",
                                        show_plot=show_all_graphs)
        Plotter.plot_trajectory_entropy(footbot_list=faulty_bots,
                                        path=saving_graphs_file_path,
                                        title="Fault Bots Trajectory Entropy",
                                        show_plot=show_all_graphs)
        Plotter.plot_footbot_area_coverage(footbot_list=nominal_bots,
                                           path=saving_graphs_file_path,
                                           additional_title_string="Nominal",
                                           show_plot=show_all_graphs)
        Plotter.plot_footbot_area_coverage(footbot_list=faulty_bots,
                                           path=saving_graphs_file_path,
                                           additional_title_string="Fault",
                                           show_plot=show_all_graphs)
        Plotter.plot_swarm_area_coverage(main_swarm=main_swarm,
                                         path=saving_graphs_file_path,
                                         show_plot=show_all_graphs)
        Plotter.plot_distances_from_centroid(footbot_list=nominal_bots,
                                             path=saving_graphs_file_path,
                                             additional_title_string="Nominal",
                                             show_plot=show_all_graphs)
        Plotter.plot_distances_from_centroid(footbot_list=faulty_bots,
                                             path=saving_graphs_file_path,
                                             additional_title_string="Fault",
                                             show_plot=show_all_graphs)
        Plotter.plot_swarm_cohesion(footbot_list=nominal_bots,
                                    path=saving_graphs_file_path,
                                    title="Nominal Average Distance from all Bots for each bot",
                                    show_plot=show_all_graphs)
        Plotter.plot_swarm_cohesion(footbot_list=faulty_bots,
                                    path=saving_graphs_file_path,
                                    title="Fault Average Distance from all Bots for each bot",
                                    show_plot=show_all_graphs)
        Plotter.plot_neighbors(footbot_list=nominal_bots,
                               path=saving_graphs_file_path,
                               title="Nominal Bots Neighbors",
                               show_plot=show_all_graphs)
        Plotter.plot_neighbors(footbot_list=faulty_bots,
                               path=saving_graphs_file_path,
                               title="Fault Bots Neighbors",
                               show_plot=show_all_graphs)
        Plotter.plot_swarm_speed(footbot_list=main_swarm,
                                 path=saving_graphs_file_path,
                                 show_plot=False)
        Plotter.plot_faulty_robots(footbot_list=main_swarm.list_of_footbots,
                                   path=saving_graphs_file_path,
                                   title="Number of Faulty Bots",
                                   show_plot=show_all_graphs)

    @staticmethod
    def plot_foraging_features(nominal_bots: List[FootBot],
                               faulty_bots: List[FootBot],
                               saving_graphs_file_path,
                               show_all_graphs: bool = True):
        Plotter.plot_exploring_unsuccessfully_time_series(footbot_list=nominal_bots,
                                                          path=saving_graphs_file_path,
                                                          title="Nominal Bot searching time",
                                                          show_plot=show_all_graphs)
        Plotter.plot_exploring_unsuccessfully_time_series(footbot_list=faulty_bots,
                                                          path=saving_graphs_file_path,
                                                          title="Faulty Bot searching time",
                                                          show_plot=show_all_graphs)
        Plotter.plot_has_food_time_series(footbot_list=nominal_bots,
                                          path=saving_graphs_file_path,
                                          title="Nominal Bot carrying food",
                                          show_plot=show_all_graphs)
        Plotter.plot_has_food_time_series(footbot_list=faulty_bots,
                                          path=saving_graphs_file_path,
                                          title="Faulty Bot carrying food",
                                          show_plot=show_all_graphs)
        Plotter.plot_state_time_series(footbot_list=nominal_bots,
                                       path=saving_graphs_file_path,
                                       title="Nominal Bot states",
                                       show_plot=show_all_graphs)
        Plotter.plot_state_time_series(footbot_list=faulty_bots,
                                       path=saving_graphs_file_path,
                                       title="Faulty Bot states",
                                       show_plot=show_all_graphs)
        Plotter.plot_time_rested_time_series(footbot_list=nominal_bots,
                                             path=saving_graphs_file_path,
                                             title="Nominal Bot rested time",
                                             show_plot=show_all_graphs)
        Plotter.plot_time_rested_time_series(footbot_list=faulty_bots,
                                             path=saving_graphs_file_path,
                                             title="Faulty Bot rested time",
                                             show_plot=show_all_graphs)
        Plotter.plot_total_food_time_series(footbot_list=nominal_bots,
                                            path=saving_graphs_file_path,
                                            title="Nominal Bot total food",
                                            show_plot=show_all_graphs)
        Plotter.plot_total_food_time_series(footbot_list=faulty_bots,
                                            path=saving_graphs_file_path,
                                            title="Faulty Bot total food",
                                            show_plot=show_all_graphs)
        Plotter.plot_time_searching_for_nest_time_series(footbot_list=nominal_bots,
                                                         path=saving_graphs_file_path,
                                                         title="Nominal Bot rested time",
                                                         show_plot=show_all_graphs)
        Plotter.plot_time_searching_for_nest_time_series(footbot_list=faulty_bots,
                                                         path=saving_graphs_file_path,
                                                         title="Faulty Bot rested time",
                                                         show_plot=show_all_graphs)

    @staticmethod
    def main_foraging(par_task_name: str,
                      feature_set_features_list: List[str],
                      file_number: int,
                      show_all_graphs: bool = True):
        saving_graphs_file_path = Plotter.make_folder_from_json(par_task_name=par_task_name, file_number=file_number)

        footbots_list, main_swarm = Plotter.build_foraging_swarm(par_task_name=par_task_name,
                                                                 feature_set_features_list=feature_set_features_list,
                                                                 file_number=file_number)

        faulty_bots, nominal_bots = Plotter.divide_flocks(footbots_list=footbots_list)

        Plotter.plot_common_features(nominal_bots=nominal_bots,
                                     faulty_bots=faulty_bots,
                                     main_swarm=main_swarm,
                                     saving_graphs_file_path=saving_graphs_file_path,
                                     show_all_graphs=show_all_graphs)

        Plotter.plot_foraging_features(nominal_bots=nominal_bots,
                                       faulty_bots=faulty_bots,
                                       saving_graphs_file_path=saving_graphs_file_path,
                                       show_all_graphs=show_all_graphs)

    @staticmethod
    def plot_all_cached_swarm_in_directory(task_name: str,
                                           show_images_in_new_window: bool):
        folder = Parser.return_cached_swarm_directory_path(experiment_name=task_name)
        root = Parser.get_project_root()
        images_path = join(root, 'images')
        for file in listdir(folder):
            file_path = join(folder, file)
            with open(file_path, 'rb') as f:
                swarm = pickle.load(f)
            print('loaded {}'.format(file))
            folder_path = Plotter.make_folder_from_complete_file_path(join(images_path, file))
            faulty_bots, nominal_bots = Plotter.divide_flocks(swarm.list_of_footbots)
            Plotter.plot_traj_and_fault_in_separate_window(nominal_bots=nominal_bots,
                                                           faulty_bots=faulty_bots,
                                                           saving_path=folder_path,
                                                           main_swarm=swarm,
                                                           title=file.replace('.pkl', ''),
                                                           show_graphs=False)
            if show_images_in_new_window:
                for image_file in listdir(folder_path):
                    image_path = join(folder_path, image_file)
                    img = Image.open(image_path)
                    img.show()

    @staticmethod
    def plot_from_json_cached_swarm(par_task_name: str,
                                    file_number: int,
                                    title: str,
                                    show_all_graphs: bool = True):
        saving_graphs_file_path = Plotter.make_folder_from_json(par_task_name=par_task_name, file_number=file_number)

        footbots_list, main_swarm = Plotter.load_generic_swarm_from_json(par_task_name=par_task_name,
                                                                         file_number=file_number)

        faulty_bots, nominal_bots = Plotter.divide_flocks(footbots_list=footbots_list)

        Plotter.plot_traj_and_fault_in_separate_window(nominal_bots=nominal_bots,
                                                       faulty_bots=faulty_bots,
                                                       saving_path=saving_graphs_file_path,
                                                       main_swarm=main_swarm,
                                                       title=title,
                                                       show_graphs=show_all_graphs)

    @staticmethod
    def load_and_plot_univ_multiv_ts_from_json(par_task_name: str,
                                               file_number: int,
                                               show_all_graphs: bool = True):
        saving_graphs_file_path = Plotter.make_folder_from_json(par_task_name=par_task_name, file_number=file_number)

        footbots_list, main_swarm = Plotter.load_generic_swarm_from_json(par_task_name=par_task_name,
                                                                         file_number=file_number)

        Plotter.plot_uinv_multiv_ts(saving_path=saving_graphs_file_path,
                                    main_swarm=main_swarm,
                                    show_graphs=show_all_graphs)

    @staticmethod
    def load_generic_swarm_from_json(par_task_name: str,
                                     file_number: int) -> Tuple[List[FootBot], Swarm]:
        filename = Parser.read_filename(task_name=par_task_name,
                                        file_number=file_number)
        path = Parser.return_cached_swarm_directory_path(experiment_name=par_task_name)
        path = join(path, filename)
        with open(path, 'rb') as f:
            swarm = pickle.load(f)
        print('loaded {}'.format(filename))
        return swarm.list_of_footbots, swarm

    @staticmethod
    def main_dispersion(par_task_name: str,
                        feature_set_features_list: List[str],
                        file_number: int,
                        show_all_graphs: bool = True):
        saving_graphs_file_path = Plotter.make_folder_from_json(par_task_name=par_task_name, file_number=file_number)

        footbots_list, main_swarm = Plotter.build_generic_swarm(par_task_name=par_task_name,
                                                                feature_set_features_list=feature_set_features_list,
                                                                file_number=file_number)

        faulty_bots, nominal_bots = Plotter.divide_flocks(footbots_list=footbots_list)

        Plotter.plot_common_features(nominal_bots=nominal_bots,
                                     faulty_bots=faulty_bots,
                                     main_swarm=main_swarm,
                                     saving_graphs_file_path=saving_graphs_file_path,
                                     show_all_graphs=show_all_graphs)

    @staticmethod
    def plot_time_performanca_comparison_graphs():
        data_dict = Parser.open_time_performance_json_file()
        for task in data_dict.keys():
            feature_sets = []
            feature_sets_sizes = []
            train_times = []
            data_times = []
            F1_scores = []
            for feature_set in data_dict[task].keys():
                feature_sets.append(feature_set)
                feature_sets_sizes.append(Parser.compute_feature_set_size(feature_set_name=feature_set))
                train_times.append(data_dict[task][feature_set]['Train_time'])
                data_times.append(data_dict[task][feature_set]['Data_time'])
                F1_scores.append(data_dict[task][feature_set]['F1'])
            data_ordered_by_size = sorted(zip(feature_sets,
                                              feature_sets_sizes,
                                              train_times,
                                              data_times,
                                              F1_scores), key=lambda x: x[1])
            title = '{} Performances'.format(task)
            fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(15, 15))
            fig.suptitle(title)
            ax1.plot([_[2] for _ in data_ordered_by_size])
            ax1.set_ylabel('Train seconds')
            ax2.plot([_[3] for _ in data_ordered_by_size])
            ax2.set_ylabel('Dataset Seconds')
            ax3.plot([_[4] for _ in data_ordered_by_size])
            ax3.set_ylabel('F1')
            plt.xticks(ticks=range(len(feature_sets)),
                       labels=[_[0] for _ in data_ordered_by_size],
                       rotation='vertical')

            plt.show()
            plt.close(fig)


if __name__ == "__main__":
    Plotter.plot_time_performanca_comparison_graphs()
