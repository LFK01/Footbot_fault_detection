import matplotlib.pyplot as plt


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
    def plot_trajectories(swarm: list) -> None:
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
        plt.show()

    @staticmethod
    def plot_speeds(swarm) -> None:
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
        plt.show()

    @staticmethod
    def plot_neighbors(swarm) -> None:
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
        plt.show()
