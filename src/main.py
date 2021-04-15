from src.utils.Parser import Parser
from src.utils.Plotter import Plotter
from src.Classes.Swarm import Swarm

if __name__ == "__main__":
    neighborhood_radius = Parser.read_neighborhood_radius()
    time_window_size = Parser.read_time_window()
    filename = Parser.read_filename(3)
    footbots_list = Parser.create_swarm(filename=filename,
                                        neighborhood_radius=neighborhood_radius,
                                        time_window_size=time_window_size)
    # noinspection PyTypeChecker
    swarm = Swarm(footbots_list)

    # noinspection PyTypeChecker
    Plotter.plot_neighbors(footbots_list)
    # noinspection PyTypeChecker
    Plotter.plot_speeds(footbots_list)
    # noinspection PyTypeChecker
    Plotter.plot_trajectories(footbots_list, swarm)
    # noinspection PyTypeChecker
    Plotter.plot_cumulative_traversed_distance(footbots_list)
    # noinspection PyTypeChecker
    Plotter.plot_swarm_cohesion(footbots_list)
    # noinspection PyTypeChecker
    Plotter.plot_faulty_robots(footbots_list)
    # noinspection PyTypeChecker
    Plotter.plot_swarm_speed(swarm)
    # noinspection PyTypeChecker
    Plotter.plot_distances_from_centroid(footbots_list)
