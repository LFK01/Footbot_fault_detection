from src.utils.Parser import Parser
from src.utils.Plotter import Plotter


if __name__ == "__main__":
    neighborhood_radius = Parser.read_neighborhood_radius()
    time_window_size = Parser.read_time_window()
    swarm = Parser.create_swarm('positions.csv',
                                neighborhood_radius=neighborhood_radius,
                                time_window_size=time_window_size)
    Plotter.plot_neighbors(swarm)
    Plotter.plot_speeds(swarm)
    Plotter.plot_trajectories(swarm)
    print('')
