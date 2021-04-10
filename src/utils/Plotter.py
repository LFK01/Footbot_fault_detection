import matplotlib.pyplot as plt
from src.Classes.FootBot import FootBot


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
    def plot_trajectories(swarm: list[FootBot]) -> None:
        """
        Method to plot trajectories of each robot. Shows the plot.

        Parameters
        ----------
        swarm : list
            List of FootBot instances
        """

        plt.figure()
        for bot in swarm:
            pos_x = [pos[0] for pos in bot.single_robot_positions]
            pos_y = [pos[1] for pos in bot.single_robot_positions]
            plt.scatter(pos_x, pos_y, s=0.1)
        plt.xlabel("X")
        plt.ylabel("Y")
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
        plt.ylabel("Traversed Distance")
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
        plt.show()
