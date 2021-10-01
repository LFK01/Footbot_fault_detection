import pickle

from src.Utils.data_utils.BotDataset import BotDataset
from src.Utils.models.GbModel import GbModel


def execute_training():
    with open('../cached_files/cached_datasets/bot_datasets_down_sampled_20_30-09-2021_18-20.pkl', 'rb') as input_file:
        data_wizard_datasets: list[BotDataset] = pickle.load(input_file)

    model = GbModel(datasets=data_wizard_datasets)

    model.train()


def plot_trained_model_performances():
    with open('../cached_files/cached_datasets/bot_datasets_down_sampled_20_30-09-2021_18-20.pkl', 'rb') as input_file:
        data_wizard_datasets: list[BotDataset] = pickle.load(input_file)

    model = GbModel(datasets=data_wizard_datasets)

    model.saved_train_plot_performances()


if __name__ == "__main__":
    execute_training()
