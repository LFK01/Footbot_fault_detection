import os
from datetime import datetime
import pickle

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
    def plot_trajectories(footbot_list: list[FootBot],
                          swarm: Swarm,
                          plot_swarm: bool,
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

        fig = plt.figure()
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
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(title)
        plt.legend(loc="upper right", markerscale=18)
        path = os.path.join(path, title.replace(" ", "_"))
        plt.savefig(path)
        if show_plot:
            plt.show()
        plt.close(fig)

    @staticmethod
    def plot_trajectory_entropy(footbot_list: list[FootBot],
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
        for bot in footbot_list:
            plt.plot(bot.positions_entropy, alpha=0.5)
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
    def plot_state_time_series(footbot_list: list[FootBot],
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
    def plot_time_rested_time_series(footbot_list: list[FootBot],
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
    def plot_has_food_time_series(footbot_list: list[FootBot],
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
    def plot_total_food_time_series(footbot_list: list[FootBot],
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
    def plot_time_searching_for_nest_time_series(footbot_list: list[FootBot],
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
    def plot_exploring_unsuccessfully_time_series(footbot_list: list[FootBot],
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
    def plot_speeds(footbot_list: list[FootBot],
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
        plt.ylabel("Traversed Distance")
        plt.title(title)
        if path != "":
            path += "/"
        plt.savefig(path + title.replace(" ", "_"))
        if show_plot:
            plt.show()
        plt.close(fig)

    @staticmethod
    def plot_cumulative_traversed_distance(footbot_list: list[FootBot],
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
    def plot_neighbors(footbot_list: list[FootBot],
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
    def plot_faulty_robots(footbot_list: list[FootBot],
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
    def plot_swarm_cohesion(footbot_list: list[FootBot],
                            path: str = "",
                            title: str = "Average distance from all bots for each bot",
                            show_plot: bool = True) -> None:
        """
        Method to plot cumulative of each robot. Shows the plot.

        Parameters
        ----------
        footbot_list : list[FootBot]
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
                         title: str = "Instantaneous Traversed Distance of the cluster",
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
    def plot_distances_from_centroid(footbot_list: list[FootBot],
                                     path: str = "",
                                     title: str = "Distance from centroid for each bot",
                                     show_plot: bool = True) -> None:
        """
        Method to plot distances from centroid. Shows the plot.

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
            plt.plot(bot.distance_from_centroid_time_series)
        plt.xlabel("Timestep")
        plt.ylabel("Distance from Centroid")
        plt.title("Distance from centroid for each bot")
        plt.show()
        plt.close(fig)

        fig = plt.figure()
        for bot in footbot_list:
            plt.plot(bot.cumulative_distance_from_centroid_time_series)
        plt.xlabel("Timestep")
        plt.ylabel("Distance from Centroid")
        plt.title(title)
        if path != "":
            path += "/"
        plt.savefig(path + title.replace(" ", "_"))
        if show_plot:
            plt.show()
        plt.close(fig)

    @staticmethod
    def plot_footbot_area_coverage(footbot_list: list[FootBot],
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


def make_folder(par_task_name: str, file_number: int) -> str:
    new_folder_name = Parser.read_filename(task_name=par_task_name,
                                           file_number=file_number).split('/')[-1].split('.')[0]
    # find project root
    root = Parser.get_project_root()
    saving_graphs_file_path = os.path.join(root, 'images', new_folder_name)

    if os.path.exists(saving_graphs_file_path):
        saving_graphs_file_path = saving_graphs_file_path + '_' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    os.makedirs(saving_graphs_file_path)

    return saving_graphs_file_path


def divide_flocks(footbots_list: list[FootBot]) -> tuple[list[FootBot], list[FootBot]]:
    faulty_bots = [bot for bot in footbots_list if any(bot.fault_time_series)]
    nominal_bots = [bot for bot in footbots_list if not any(bot.fault_time_series)]

    return faulty_bots, nominal_bots


def build_generic_swarm(par_task_name: str, file_number: int):
    neighborhood_radius = Parser.read_neighborhood_radius()
    time_window_size = Parser.read_time_window()
    file = Parser.read_filename(task_name=par_task_name, file_number=file_number)
    timesteps = Parser.retrieve_timesteps_series_from_dataframe(
        df_footbot_positions=Parser.open_pandas_dataframe(filename=file,
                                                          task_name=task_name)
    )
    footbots_list = Parser.create_generic_swarm(task_name=task_name,
                                                filename=file,
                                                neighborhood_radius=neighborhood_radius,
                                                time_window_size=time_window_size)
    return footbots_list, Swarm(timesteps=timesteps,
                                swarm=footbots_list)


def build_foraging_swarm(par_task_name: str, file_number: int):
    neighborhood_radius = Parser.read_neighborhood_radius()
    time_window_size = Parser.read_time_window()
    file = Parser.read_filename(task_name=par_task_name, file_number=file_number)
    timesteps = Parser.retrieve_timesteps_series_from_dataframe(
        df_footbot_positions=Parser.open_pandas_dataframe(filename=file,
                                                          task_name=par_task_name)
    )

    footbots_list = Parser.create_foraging_swarm(filename=file,
                                                 neighborhood_radius=neighborhood_radius,
                                                 time_window_size=time_window_size)
    root = Parser.get_project_root()
    path = os.path.join(root, 'cached_files')
    with open(os.path.join(path, 'foraging_footbot_list' +
                                 datetime.now().strftime('%d-%m-%Y_%H-%M') +
                                 '70_bots_800sec.pkl'),
              'wb') as output_file:
        pickle.dump(footbots_list, output_file)

    return footbots_list, Swarm(timesteps=timesteps,
                                swarm=footbots_list)


def plot_common_features(nominal_bots: list[FootBot],
                         faulty_bots: list[FootBot],
                         main_swarm: Swarm,
                         saving_graphs_file_path,
                         show_all_graphs: bool = True):
    Plotter.plot_trajectories(footbot_list=nominal_bots,
                              swarm=main_swarm,
                              plot_swarm=False,
                              path=saving_graphs_file_path,
                              title="Nominal Bots Trajectories",
                              show_plot=show_all_graphs)
    Plotter.plot_trajectories(footbot_list=faulty_bots,
                              swarm=main_swarm,
                              plot_swarm=False,
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

    Plotter.plot_faulty_robots(footbot_list=main_swarm.list_of_footbots,
                               path=saving_graphs_file_path,
                               title="Number of Faulty Bots",
                               show_plot=show_all_graphs)


def plot_foraging_features(nominal_bots: list[FootBot],
                           faulty_bots: list[FootBot],
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


def main_foraging(par_task_name: str,
                  file_number: int,
                  show_all_graphs: bool = True):
    saving_graphs_file_path = make_folder(par_task_name=par_task_name, file_number=file_number)

    footbots_list, main_swarm = build_foraging_swarm(par_task_name=par_task_name,
                                                     file_number=file_number)

    faulty_bots, nominal_bots = divide_flocks(footbots_list=footbots_list)

    plot_common_features(nominal_bots=nominal_bots,
                         faulty_bots=faulty_bots,
                         main_swarm=main_swarm,
                         saving_graphs_file_path=saving_graphs_file_path,
                         show_all_graphs=show_all_graphs)

    plot_foraging_features(nominal_bots=nominal_bots,
                           faulty_bots=faulty_bots,
                           saving_graphs_file_path=saving_graphs_file_path)


def main_dispersion(par_task_name: str,
                    file_number: int,
                    show_all_graphs: bool = True):
    saving_graphs_file_path = make_folder(par_task_name=par_task_name, file_number=file_number)

    footbots_list, main_swarm = build_generic_swarm(par_task_name=par_task_name,
                                                    file_number=file_number)

    faulty_bots, nominal_bots = divide_flocks(footbots_list=footbots_list)

    plot_common_features(nominal_bots=nominal_bots,
                         faulty_bots=faulty_bots,
                         main_swarm=main_swarm,
                         saving_graphs_file_path=saving_graphs_file_path,
                         show_all_graphs=show_all_graphs)


def plot_model_performances():
    x_values = [0, 1, 2, 3, 4, 5]
    y_values = [0.8700, 0.9840, 0.7904, 0.9994, 0.9920, 0.8335]
    bot_labels = ['Bot 0', 'Bot 1', 'Bot 3', 'Bot 4', 'Bot 7', 'Bot 8']

    plt.figure()
    plt.scatter(x=x_values, y=y_values, label='single bot', alpha=0.7)
    for x_val, y_val in zip(x_values, y_values):
        plt.text(x_val + 0.03, y_val + 0.03, str(y_val)[:4])
    plt.xticks(ticks=x_values, labels=bot_labels)
    plt.axis([-0.5, 5.5, -0.1, 1.1])
    plt.ylabel("Performance")
    plt.title("Gradient Boosting Performance")
    plt.hlines(0.9427457141610229, xmin=0, xmax=5, color='r', linestyles='dashed', label='merged bots')
    plt.text(5, 0.97, str(0.94))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    task_name = 'WARE'
    main_dispersion(par_task_name=task_name,
                    file_number=1,
                    show_all_graphs=False)
