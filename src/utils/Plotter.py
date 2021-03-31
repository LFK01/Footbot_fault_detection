import matplotlib.pyplot as plt


class Plotter:
    def __init__(self):
        pass

    @staticmethod
    def plot_trajectories(swarm):
        plt.figure()
        for bot in swarm:
            pos_x = [pos[0] for pos in bot.positions]
            pos_y = [pos[1] for pos in bot.positions]
            plt.scatter(pos_x, pos_y, s=0.1)
        plt.show()

    @staticmethod
    def plot_speeds(swarm):
        plt.figure()
        for bot in swarm:
            plt.plot(bot.speed_data)
        plt.show()
