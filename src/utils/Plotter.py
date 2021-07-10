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
    def plot_trajectory_entropy(swarm: list[FootBot]) -> None:
        """
        Method to plot the entropy of the trajectory. Shows the plot.

        Parameters
        ----------
        swarm : list
            List of FootBot instances
        """
        plt.figure()
        for bot in swarm:
            plt.plot(bot.positions_entropy, alpha=0.5)
        plt.xlabel("Timestep")
        plt.ylabel("Entropy")
        plt.title("Entropy for each bot")
        plt.show()

    @staticmethod
    def plot_state_time_series(swarm: list[FootBot]) -> None:
        """
        Method to plot state of each robot. Shows the plot.

        Parameters
        ----------
        swarm : list
            List of FootBot instances
        """
        plt.figure()
        for bot in swarm:
            plt.plot(bot.state_time_series)
        plt.xlabel("Timestep")
        plt.ylabel("State")
        plt.title("State for each bot")
        plt.text(0, 0, 'STATE = Resting')
        plt.text(0, 1, 'STATE = Exploring')
        plt.text(0, 2, 'STATE = Returning to nest')
        plt.show()

    @staticmethod
    def plot_time_rested_time_series(swarm: list[FootBot]) -> None:
        """
        Method to plot rested_time of each robot. Shows the plot.

        Parameters
        ----------
        swarm : list
            List of FootBot instances
        """
        plt.figure()
        for bot in swarm:
            plt.plot(bot.TimeRested_time_series)
        plt.xlabel("Timestep")
        plt.ylabel("Rested Time")
        plt.title("Rested Time for each bot")
        plt.show()

    @staticmethod
    def plot_has_food_time_series(swarm: list[FootBot]) -> None:
        """
        Method to plot if each robot has food. Shows the plot.

        Parameters
        ----------
        swarm : list
            List of FootBot instances
        """
        plt.figure()
        for bot in swarm:
            plt.plot(bot.HasFood_time_series)
        plt.xlabel("Timestep")
        plt.ylabel("Does it have food?")
        plt.title("Has food info for each bot")
        plt.show()

    @staticmethod
    def plot_total_food_time_series(swarm: list[FootBot]) -> None:
        """
        Method to plot total food collected from each bot. Shows the plot.

        Parameters
        ----------
        swarm : list
            List of FootBot instances
        """
        plt.figure()
        for bot in swarm:
            plt.plot(bot.TotalFood_time_series)
        plt.xlabel("Timestep")
        plt.ylabel("Collected food")
        plt.title("Collected food for each bot")
        plt.show()

    @staticmethod
    def plot_time_searching_for_nest_time_series(swarm: list[FootBot]) -> None:
        """
        Method to plot if each robot has food. Shows the plot.

        Parameters
        ----------
        swarm : list
            List of FootBot instances
        """
        plt.figure()
        for bot in swarm:
            plt.plot(bot.TimeSearchingForNest_time_series)
        plt.xlabel("Timestep")
        plt.ylabel("Nest Time")
        plt.title("Time to search space in nest")
        plt.show()

    @staticmethod
    def plot_exploring_unsuccessfully_time_series(swarm: list[FootBot]) -> None:
        """
        Method to plot if each robot has food. Shows the plot.

        Parameters
        ----------
        swarm : list
            List of FootBot instances
        """
        plt.figure()
        for bot in swarm:
            plt.plot(bot.TimeExploringUnsuccessfully_time_series)
        plt.xlabel("Timestep")
        plt.ylabel("Exploration time")
        plt.title("Unsuccessful exploration time")
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
        swarm : list[FootBot]
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


def build_swarm(file_number: int):
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

    Plotter.plot_trajectories(footbots_list, main_swarm)
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
    footbots_list, main_swarm = build_swarm(1)

    Plotter.plot_trajectories(footbots_list, main_swarm)
    Plotter.plot_speeds(footbots_list)
    Plotter.plot_cumulative_traversed_distance(footbots_list)

    Plotter.plot_swarm_cohesion(footbots_list)
    Plotter.plot_neighbors(footbots_list)

    Plotter.plot_faulty_robots(footbots_list)


def main_dispersion():
    footbots_list, main_swarm = build_swarm(7)

    Plotter.plot_trajectories(footbots_list, main_swarm)
    Plotter.plot_speeds(footbots_list)
    Plotter.plot_cumulative_traversed_distance(footbots_list)
    Plotter.plot_trajectory_entropy(footbots_list)

    Plotter.plot_swarm_cohesion(footbots_list)
    Plotter.plot_neighbors(footbots_list)

    Plotter.plot_faulty_robots(footbots_list)


def plot_model_performances():

    x_values = [0, 1, 2, 3, 4, 5]
    y_values = [0.8700, 0.9840, 0.7904, 0.9994, 0.9920, 0.8335]
    bot_labels = ['Bot 0', 'Bot 1', 'Bot 3', 'Bot 4', 'Bot 7', 'Bot 8']

    plt.figure()
    plt.scatter(x=x_values, y=y_values, label='single bot', alpha=0.7)
    for x_val, y_val in zip(x_values, y_values):
        plt.text(x_val+0.03, y_val+0.03, str(y_val)[:4])
    plt.xticks(ticks=x_values, labels=bot_labels)
    plt.axis([-0.5, 5.5, -0.1, 1.1])
    plt.ylabel("Performance")
    plt.title("Gradient Boosting Performance")
    plt.hlines(0.9427457141610229, xmin=0, xmax=5, color='r', linestyles='dashed', label='merged bots')
    plt.text(5, 0.97, str(0.94))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main_dispersion()

