from src.utils.Parser import Parser
from src.utils.Plotter import Plotter


if __name__ == "__main__":
    swarm = Parser.create_swarm('positions.csv')
    Plotter.plot_speeds(swarm)
    print('')
