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
    def plot_trajectories(footbot_list: list[FootBot], swarm: Swarm) -> None:
        """
        Method to plot trajectories of each robot. Shows the plot.

        Parameters
        ----------
        swarm: Swarm
            Swarm object to retrieve cluster features
        footbot_list : list
            List of FootBot instances
        """

        plt.figure()
        for bot in footbot_list:
            plt.scatter(
                [pos[0] for pos in bot.single_robot_positions],
                [pos[1] for pos in bot.single_robot_positions],
                s=0.1)
        plt.scatter(
            [pos[0] for pos in swarm.trajectory],
            [pos[1] for pos in swarm.trajectory],
            facecolors='none', edgecolors='r', s=0.2
        )
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Trajectory of each bot and centroid of cluster")
        plt.show()

    @staticmethod
    def plot_speeds(swarm: list[FootBot]) -> None:
        """
        Method to plot speeds of each robot. Shows the plot.

        Parameters
        ----------
        swarm : list
            List of FootBot instances
        """
        plt.figure()
        for bot in swarm:
            plt.plot(bot.traversed_distance_time_series)
        plt.xlabel("Timestep")
        plt.ylabel("Traversed Distance")
        plt.title("Instantaneous Traversed Distance for each bot")
        plt.show()

    @staticmethod
    def plot_cumulative_traversed_distance(swarm: list[FootBot]) -> None:
        """
        Method to plot cumulative of each robot. Shows the plot.

        Parameters
        ----------
        swarm : list
            List of FootBot instances
        """
        plt.figure()
        for bot in swarm:
            plt.plot(bot.cumulative_traversed_distance)
        plt.xlabel("Timestep")
        plt.ylabel("Cumulative Traversed Distance")
        plt.title("Cumulative Traversed Distance for each bot")
        plt.show()

    @staticmethod
    def plot_neighbors(swarm: list[FootBot]) -> None:
        """
        Method to plot neighbors of each robot. Shows the plot.

        Parameters
        ----------
        swarm : list
            List of FootBot instances
        """
        plt.figure()
        for bot in swarm:
            plt.plot(bot.neighbors_time_series)
        plt.xlabel("Timestep")
        plt.ylabel("Number of Neighbors")
        plt.title("Number of Neighbors for each bot")
        plt.show()

    @staticmethod
    def plot_faulty_robots(swarm: list[FootBot]) -> None:
        """
        Method to plot neighbors of each robot. Shows the plot.

        Parameters
        ----------
        swarm : list
            List of FootBot instances
        """
        cumulative_faulty_bots = []
        plt.figure()
        for timestep in range(len(swarm[0].fault_time_series)):
            cumulative_faulty_bots.append(
                sum(bot.fault_time_series[timestep] for bot in swarm)
            )
        plt.plot(cumulative_faulty_bots)
        plt.xlabel("Timestep")
        plt.ylabel("Number of faulty bots")
        plt.title("Number of faulty bots")
        plt.show()

    @staticmethod
    def plot_swarm_cohesion(swarm: list[FootBot]) -> None:
        """
        Method to plot cumulative of each robot. Shows the plot.

        Parameters
        ----------
        swarm : list
            List of FootBot instances
        """
        plt.figure()
        for bot in swarm:
            plt.plot(bot.swarm_cohesion_time_series)
        plt.xlabel("Timestep")
        plt.ylabel("Average Distance")
        plt.title("Average distance from all bots for each bot")
        plt.show()

    @staticmethod
    def plot_swarm_speed(swarm: Swarm):
        """
        Method to plot trajectory of swarm. Shows the plot.

        Parameters
        ----------
        swarm : Swarm
            instance of the swarm of bots
        """
        plt.figure()
        plt.plot(swarm.traversed_distance_time_series)
        plt.xlabel("timestep")
        plt.ylabel("Swarm speed")
        plt.title("Instantaneous Traversed Distance of the cluster")
        plt.show()

    @staticmethod
    def plot_distances_from_centroid(swarm: list[FootBot]) -> None:
        """
        Method to plot distances from centroid. Shows the plot.

        Parameters
        ----------
        swarm : list
            List of FootBot instances
        """
        plt.figure()
        for bot in swarm:
            plt.plot(bot.distance_from_centroid_time_series)
        plt.xlabel("Timestep")
        plt.ylabel("Distance from Centroid")
        plt.title("Distance from centroid for each bot")
        plt.show()

        plt.figure()
        for bot in swarm:
            plt.plot(bot.cumulative_distance_from_centroid_time_series)
        plt.xlabel("Timestep")
        plt.ylabel("Distance from Centroid")
        plt.title("Cumulative Distance from centroid for each bot")
        plt.show()


if __name__ == "__main__":
    neighborhood_radius = Parser.read_neighborhood_radius()
    time_window_size = Parser.read_time_window()
    file = Parser.read_filename(1)
    footbots_list = Parser.create_swarm(filename=file,
                                        neighborhood_radius=neighborhood_radius,
                                        time_window_size=time_window_size)
    # noinspection PyTypeChecker
    main_swarm = Swarm(footbots_list)

    # noinspection PyTypeChecker
    Plotter.plot_trajectories(footbots_list, main_swarm)
    # noinspection PyTypeChecker
    Plotter.plot_speeds(footbots_list)
    # noinspection PyTypeChecker
    Plotter.plot_neighbors(footbots_list)
    # noinspection PyTypeChecker
    Plotter.plot_cumulative_traversed_distance(footbots_list)
    # noinspection PyTypeChecker
    Plotter.plot_swarm_cohesion(footbots_list)
    # noinspection PyTypeChecker
    Plotter.plot_distances_from_centroid(footbots_list)
    # noinspection PyTypeChecker
    Plotter.plot_swarm_speed(main_swarm)
    # noinspection PyTypeChecker
    Plotter.plot_faulty_robots(footbots_list)

