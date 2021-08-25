import pickle
from src.utils.models.ConvLstmModel import ConvLstmModel
from src.utils.models.GbModel import GbModel
from src.utils.data_utils.BotDataset import BotDataset


if __name__ == "__main__":
    with open('../cached_objects/bot_data_wizard_normalized.pkl', 'rb') as input_file:
        data_wizard_datasets: list[BotDataset] = pickle.load(input_file)

    model = GbModel(datasets=data_wizard_datasets)
    model.train()

    print(' ')
