import pickle

from src.utils.data_utils.BotDataset import BotDataset
from src.utils.models.GbModel import GbModel

if __name__ == "__main__":
    with open('../cached_objects/bot_datasets_down_sampled.pkl', 'rb') as input_file:
        data_wizard_datasets: list[BotDataset] = pickle.load(input_file)

    model = GbModel(datasets=data_wizard_datasets)

    model.train()

    print(' ')
