from src.utils.Parser import Parser
from src.utils.Plotter import Plotter


if __name__ == "__main__":
    neighborhood_radius = Parser.read_neighborhood_radius()
    swarm = Parser.create_swarm('positions.csv', neighborhood_radius)
    Plotter.plot_neighbors(swarm)
    print('')
