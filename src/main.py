import pandas as pd
from FootBot import FootBot

if __name__ == "__main__":
    footbot_swarm = []

    df_footbot_positions = pd.read_csv('../csv_log_files/Log_positions_diffusion_Sat_Mar_27_17-36-54_2021.csv')
    footbots_unique_ids = df_footbot_positions['ID'].unique()
    for footbot_id in footbots_unique_ids:
        new_footbot = FootBot(footbot_id)
        footbot_swarm.append(new_footbot)
        positions = df_footbot_positions[df_footbot_positions['ID'] == footbot_id][['PosX', 'PosY']]
        new_footbot.add_list_positions(positions.values.tolist())
