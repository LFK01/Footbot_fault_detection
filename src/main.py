import matplotlib.pyplot as plt
from src.File_parsing.Parser import Parser
if __name__ == "__main__":
    swarm = Parser.create_swarm('positions.csv')

    plt.figure()
    for bot in swarm:
        pos_x = [pos[0] for pos in bot.positions]
        pos_y = [pos[1] for pos in bot.positions]
        plt.plot(pos_x, pos_y)
    plt.show()

