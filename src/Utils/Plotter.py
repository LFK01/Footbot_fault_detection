import os
import datetime

import matplotlib.pyplot as plt
from src.Classes.FootBot import FootBot
from src.Classes.Swarm import Swarm
from src.utils.Parser import Parser


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
                          title: str = "Trajectory of each bot and centroid of cluster") -> None:
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
        """

        plt.figure()
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
        if path != "":
            path += "/"
        plt.savefig(path + title.replace(" ", "_"))
        plt.show()

    @staticmethod
    def plot_trajectory_entropy(footbot_list: list[FootBot],
                                path: str = "",
                                title: str = 'Entropy for each bot') -> None:
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
        """
        plt.figure()
        for bot in footbot_list:
            plt.plot(bot.positions_entropy, alpha=0.5)
        plt.xlabel("Timestep")
        plt.ylabel("Entropy")
        plt.title(title)
        if path != "":
            path += "/"
        plt.savefig(path + title.replace(" ", "_"))
        plt.show()

    @staticmethod
    def plot_state_time_series(footbot_list: list[FootBot],
                               path: str = "",
                               title: str = "State for each bot") -> None:
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
        """
        plt.figure()
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
        plt.show()

    @staticmethod
    def plot_time_rested_time_series(footbot_list: list[FootBot],
                                     path: str = "",
                                     title: str = "Rested Time for each bot") -> None:
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
        """
        plt.figure()
        for bot in footbot_list:
            plt.plot(bot.TimeRested_time_series)
        plt.xlabel("Timestep")
        plt.ylabel("Rested Time")
        plt.title(title)
        if path != "":
            path += "/"
        plt.savefig(path + title.replace(" ", "_"))
        plt.show()

    @staticmethod
    def plot_has_food_time_series(footbot_list: list[FootBot],
                                  path: str = "",
                                  title: str = "Has food info for each bot") -> None:
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
        """
        plt.figure()
        for bot in footbot_list:
            plt.plot(bot.HasFood_time_series)
        plt.xlabel("Timestep")
        plt.ylabel("Does it have food?")
        plt.title(title)
        if path != "":
            path += "/"
        plt.savefig(path + title.replace(" ", "_"))
        plt.show()

    @staticmethod
    def plot_total_food_time_series(footbot_list: list[FootBot],
                                    path: str = "",
                                    title: str = "Collected food for each bot") -> None:
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
        """
        plt.figure()
        for bot in footbot_list:
            plt.plot(bot.TotalFood_time_series)
        plt.xlabel("Timestep")
        plt.ylabel("Collected food")
        plt.title(title)
        if path != "":
            path += "/"
        plt.savefig(path + title.replace(" ", "_"))
        plt.show()

    @staticmethod
    def plot_time_searching_for_nest_time_series(footbot_list: list[FootBot],
                                                 path: str = "",
                                                 title: str = "Time to search space in nest") -> None:
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
        """
        plt.figure()
        for bot in footbot_list:
            plt.plot(bot.TimeSearchingForNest_time_series)
        plt.xlabel("Timestep")
        plt.ylabel("Nest Time")
        plt.title(title)
        if path != "":
            path += "/"
        plt.savefig(path + title.replace(" ", "_"))
        plt.show()

    @staticmethod
    def plot_exploring_unsuccessfully_time_series(footbot_list: list[FootBot],
                                                  path: str = "",
                                                  title: str = "Time to search space in nest") -> None:
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
        """
        plt.figure()
        for bot in footbot_list:
            plt.plot(bot.TimeExploringUnsuccessfully_time_series)
        plt.xlabel("Timestep")
        plt.ylabel("Exploration time")
        plt.title(title)
        if path != "":
            path += "/"
        plt.savefig(path + title.replace(" ", "_"))
        plt.show()

    @staticmethod
    def plot_speeds(footbot_list: list[FootBot],
                    path: str = "",
                    title: str = 'Instantaneous Traversed Distance for each bot') -> None:
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
        """
        plt.figure()
        for bot in footbot_list:
            plt.plot(bot.traversed_distance_time_series)
        plt.xlabel("Timestep")
        plt.ylabel("Traversed Distance")
        plt.title(title)
        if path != "":
            path += "/"
        plt.savefig(path + title.replace(" ", "_"))
        plt.show()

    @staticmethod
    def plot_cumulative_traversed_distance(footbot_list: list[FootBot],
                                           path: str = "",
                                           title: str = 'Cumulative Traversed Distance for each bot') -> None:
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
        """
        plt.figure()
        for bot in footbot_list:
            plt.plot(bot.cumulative_traversed_distance)
        plt.xlabel("Timestep")
        plt.ylabel("Cumulative Traversed Distance")
        plt.title(title)
        if path != "":
            path += "/"
        plt.savefig(path + title.replace(" ", "_"))
        plt.show()

    @staticmethod
    def plot_neighbors(footbot_list: list[FootBot],
                       path: str = "",
                       title: str = 'Number of Neighbors for each bot') -> None:
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
        """
        plt.figure()
        for bot in footbot_list:
            plt.plot(bot.neighbors_time_series)
        plt.xlabel("Timestep")
        plt.ylabel("Number of Neighbors")
        plt.title(title)
        if path != "":
            path += "/"
        plt.savefig(path + title.replace(" ", "_"))
        plt.show()

    @staticmethod
    def plot_faulty_robots(footbot_list: list[FootBot],
                           path: str = "",
                           title: str = "Number of faulty bots") -> None:
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
        """
        cumulative_faulty_bots = []
        plt.figure()
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
        plt.show()

    @staticmethod
    def plot_swarm_cohesion(footbot_list: list[FootBot],
                            path: str = "",
                            title: str = "Average distance from all bots for each bot") -> None:
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
        """
        plt.figure()
        for bot in footbot_list:
            plt.plot(bot.swarm_cohesion_time_series)
        plt.xlabel("Timestep")
        plt.ylabel("Average Distance")
        plt.title(title)
        if path != "":
            path += "/"
        plt.savefig(path + title.replace(" ", "_"))
        plt.show()

    @staticmethod
    def plot_swarm_speed(footbot_list: Swarm,
                         path: str = "",
                         title: str = "Instantaneous Traversed Distance of the cluster"):
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
        """
        plt.figure()
        plt.plot(footbot_list.traversed_distance_time_series)
        plt.xlabel("timestep")
        plt.ylabel("Swarm speed")
        plt.title(title)
        if path != "":
            path += "/"
        plt.savefig(path + title.replace(" ", "_"))
        plt.show()

    @staticmethod
    def plot_distances_from_centroid(footbot_list: list[FootBot],
                                     path: str = "",
                                     title: str = "Distance from centroid for each bot") -> None:
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
        """
        plt.figure()
        for bot in footbot_list:
            plt.plot(bot.distance_from_centroid_time_series)
        plt.xlabel("Timestep")
        plt.ylabel("Distance from Centroid")
        plt.title("Distance from centroid for each bot")
        plt.show()

        plt.figure()
        for bot in footbot_list:
            plt.plot(bot.cumulative_distance_from_centroid_time_series)
        plt.xlabel("Timestep")
        plt.ylabel("Distance from Centroid")
        plt.title(title)
        if path != "":
            path += "/"
        plt.savefig(path + title.replace(" ", "_"))
        plt.show()

    @staticmethod
    def plot_footbot_area_coverage(footbot_list: list[FootBot],
                                   path: str = "",
                                   additional_title_string: str = "") -> None:
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
        """
        area_splits = Parser.read_area_splits()
        for percentage_index in range(len(area_splits)):
            plt.figure()
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
            plt.show()

    @staticmethod
    def plot_swarm_area_coverage(main_swarm: Swarm,
                                 path: str = "",
                                 additional_title_string: str = "") -> None:
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
        """
        for key_area_coverage in main_swarm.area_coverage.keys():
            plt.figure()
            plt.plot(main_swarm.area_coverage[key_area_coverage])
            plt.xlabel("Timestep")
            plt.ylabel("Coverage Percentage")
            plt.ylim((-0.1, 1.1))
            title = additional_title_string + " Swarm Area Coverage with " + str(key_area_coverage) + " subdivisions"
            plt.title(title)
            if path != "":
                path += "/"
            plt.savefig(path + title.strip().replace(" ", "_"))
            plt.show()


def build_flocking_swarm(file_number: int):
    neighborhood_radius = Parser.read_neighborhood_radius()
    time_window_size = Parser.read_time_window()
    file = Parser.read_filename(file_number)
    footbots_list = Parser.create_flocking_swarm(filename=file,
                                                 neighborhood_radius=neighborhood_radius,
                                                 time_window_size=time_window_size)
    return footbots_list, Swarm(footbots_list)


def build_foraging_swarm(file_number: int):
    neighborhood_radius = Parser.read_neighborhood_radius()
    time_window_size = Parser.read_time_window()
    file = Parser.read_filename(file_number)
    footbots_list = Parser.create_foraging_swarm(filename=file,
                                                 neighborhood_radius=neighborhood_radius,
                                                 time_window_size=time_window_size)
    return footbots_list, Swarm(footbots_list)


def main_foraging():
    footbots_list, main_swarm = build_foraging_swarm(6)

    Plotter.plot_trajectories(footbots_list, main_swarm, plot_swarm=True)
    Plotter.plot_speeds(footbots_list)
    Plotter.plot_cumulative_traversed_distance(footbots_list)

    Plotter.plot_state_time_series(footbots_list)
    Plotter.plot_has_food_time_series(footbots_list)
    Plotter.plot_time_rested_time_series(footbots_list)
    Plotter.plot_total_food_time_series(footbots_list)
    Plotter.plot_time_searching_for_nest_time_series(footbots_list)
    Plotter.plot_exploring_unsuccessfully_time_series(footbots_list)

    Plotter.plot_faulty_robots(footbots_list)


def main_homing():
    footbots_list, main_swarm = build_flocking_swarm(1)

    Plotter.plot_trajectories(footbots_list, main_swarm, plot_swarm=True)
    Plotter.plot_speeds(footbots_list)
    Plotter.plot_cumulative_traversed_distance(footbots_list)

    Plotter.plot_swarm_cohesion(footbots_list)
    Plotter.plot_neighbors(footbots_list)

    Plotter.plot_faulty_robots(footbots_list)


def main_dispersion(saving_graphs_file_path: str,
                    file_number: int):
    footbots_list, main_swarm = build_flocking_swarm(file_number=file_number)

    faulty_bots = [bot for bot in footbots_list if any(bot.fault_time_series)]
    nominal_bots = [bot for bot in footbots_list if not any(bot.fault_time_series)]

    Plotter.plot_trajectories(footbot_list=nominal_bots,
                              swarm=main_swarm,
                              plot_swarm=False,
                              path=saving_graphs_file_path,
                              title="Nominal Bots Trajectories")
    Plotter.plot_trajectories(footbot_list=faulty_bots,
                              swarm=main_swarm,
                              plot_swarm=False,
                              path=saving_graphs_file_path,
                              title="Fault Bots Trajectories")
    Plotter.plot_speeds(footbot_list=nominal_bots,
                        path=saving_graphs_file_path,
                        title="Nominal Bots Speeds")
    Plotter.plot_speeds(footbot_list=faulty_bots,
                        path=saving_graphs_file_path,
                        title="Fault Bots Speeds")
    Plotter.plot_cumulative_traversed_distance(footbot_list=nominal_bots,
                                               path=saving_graphs_file_path,
                                               title="Nominal Bots Cumulative Speeds")
    Plotter.plot_cumulative_traversed_distance(footbot_list=faulty_bots,
                                               path=saving_graphs_file_path,
                                               title="Fault Bots Cumulative Speeds")
    Plotter.plot_trajectory_entropy(footbot_list=nominal_bots,
                                    path=saving_graphs_file_path,
                                    title="Nominal Bots Trajectory Entropy")
    Plotter.plot_trajectory_entropy(footbot_list=faulty_bots,
                                    path=saving_graphs_file_path,
                                    title="Fault Bots Trajectory Entropy")
    Plotter.plot_footbot_area_coverage(footbot_list=nominal_bots,
                                       path=saving_graphs_file_path,
                                       additional_title_string="Nominal")
    Plotter.plot_footbot_area_coverage(footbot_list=faulty_bots,
                                       path=saving_graphs_file_path,
                                       additional_title_string="Fault")

    Plotter.plot_swarm_cohesion(footbot_list=nominal_bots,
                                path=saving_graphs_file_path,
                                title="Nominal Average Distance from all Bots for each bot")
    Plotter.plot_swarm_cohesion(footbot_list=faulty_bots,
                                path=saving_graphs_file_path,
                                title="Fault Average Distance from all Bots for each bot")
    Plotter.plot_neighbors(footbot_list=nominal_bots,
                           path=saving_graphs_file_path,
                           title="Nominal Bots Neighbors")
    Plotter.plot_neighbors(footbot_list=faulty_bots,
                           path=saving_graphs_file_path,
                           title="Fault Bots Neighbors")

    Plotter.plot_faulty_robots(footbot_list=footbots_list,
                               path=saving_graphs_file_path,
                               title="Number of Faulty Bots")


def main_warehouse(saving_graphs_file_path: str,
                   file_number: int):
    footbots_list, main_swarm = build_flocking_swarm(file_number)

    faulty_bots = [bot for bot in footbots_list if any(bot.fault_time_series)]
    nominal_bots = [bot for bot in footbots_list if not any(bot.fault_time_series)]

    Plotter.plot_trajectories(footbot_list=faulty_bots,
                              swarm=main_swarm,
                              plot_swarm=False,
                              path=saving_graphs_file_path,
                              title='Fault Bot Trajectories')
    Plotter.plot_trajectories(footbot_list=nominal_bots,
                              swarm=main_swarm,
                              plot_swarm=False,
                              path=saving_graphs_file_path,
                              title='Nominal Bot Trajectories')
    Plotter.plot_speeds(footbot_list=faulty_bots,
                        path=saving_graphs_file_path,
                        title='Fault Bot speeds')
    Plotter.plot_speeds(footbot_list=nominal_bots,
                        path=saving_graphs_file_path,
                        title='Nominal Bot speeds')
    Plotter.plot_cumulative_traversed_distance(footbot_list=faulty_bots,
                                               path=saving_graphs_file_path,
                                               title='Fault Bot Cumulative Speeds')
    Plotter.plot_cumulative_traversed_distance(footbot_list=nominal_bots,
                                               path=saving_graphs_file_path,
                                               title='Nominal Bot Cumulative Speeds')
    Plotter.plot_trajectory_entropy(footbot_list=faulty_bots,
                                    path=saving_graphs_file_path,
                                    title='Fault Bots Entropy')
    Plotter.plot_trajectory_entropy(footbot_list=nominal_bots,
                                    path=saving_graphs_file_path,
                                    title='Nominal Bots Entropy')
    Plotter.plot_footbot_area_coverage(footbot_list=faulty_bots,
                                       path=saving_graphs_file_path,
                                       additional_title_string='Fault')
    Plotter.plot_footbot_area_coverage(footbot_list=nominal_bots,
                                       path=saving_graphs_file_path,
                                       additional_title_string='Nominal')
    Plotter.plot_swarm_area_coverage(main_swarm=main_swarm,
                                     path=saving_graphs_file_path)

    Plotter.plot_swarm_cohesion(footbots_list,
                                path=saving_graphs_file_path)
    Plotter.plot_neighbors(footbot_list=faulty_bots,
                           title='Fault Bots Neighbors',
                           path=saving_graphs_file_path)
    Plotter.plot_neighbors(footbot_list=nominal_bots,
                           title='Nominal Bots Neighbors',
                           path=saving_graphs_file_path)

    Plotter.plot_faulty_robots(footbots_list,
                               path=saving_graphs_file_path)


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
    saving_folder_path = 'C:/Users/Luciano/OneDrive - Politecnico di Milano/00_TESI_directory/pdf_summaries/' \
                         'area_coverage_and_warehouse/' \
                         'nominal_fault_dispersion_100_fault_comparison_graphs'

    if os.path.exists(saving_folder_path):
        os.makedirs(saving_folder_path + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    else:
        os.makedirs(saving_folder_path)

    main_dispersion(saving_graphs_file_path=saving_folder_path,
                    file_number=4)
