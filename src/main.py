from src.utils.Parser import Parser
from src.utils.Plotter import Plotter
from src.Classes.Swarm import Swarm
from src.utils.DataTools import DataWizard

if __name__ == "__main__":
    neighborhood_radius = Parser.read_neighborhood_radius()
    time_window_size = Parser.read_time_window()
    filename = 'positions_flocking_10_bot0_lock_at100_pos4.csv'
    footbots_list = Parser.create_swarm(filename=filename,
                                        neighborhood_radius=neighborhood_radius,
                                        time_window_size=time_window_size)
    # noinspection PyTypeChecker
    swarm = Swarm(footbots_list)
    data = DataWizard.prepare_raw_input(swarm=swarm)
    target = DataWizard.prepare_target(swarm=swarm)
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
