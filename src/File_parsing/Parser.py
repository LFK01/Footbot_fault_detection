import pandas as pd
import os
from src.Classes.FootBot import FootBot


class Parser:
    def __init__(self):
        pass

    @staticmethod
    def create_swarm(filename):
        footbot_swarm = []
        cwd = os.getcwd()
        print(cwd)
        df_footbot_positions = pd.read_csv('../csv_log_files/' + filename)
        footbots_unique_ids = df_footbot_positions['ID'].unique()
        for footbot_id in footbots_unique_ids:
            new_footbot = FootBot(footbot_id)
            footbot_swarm.append(new_footbot)
            positions = df_footbot_positions[df_footbot_positions['ID'] == footbot_id][['PosX', 'PosY']]
            new_footbot.add_list_positions(positions.values.tolist())

        return footbot_swarm
