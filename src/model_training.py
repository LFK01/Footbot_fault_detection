from os.path import join
from pickle import load

from src.Utils.Parser import Parser
from src.Utils.data_utils.datasets_classes.GeneralDataset import GeneralDataset
from src.Utils.models.SingleBotGbModel import SingleBotGbModel
from src.Utils.models.ShuffledBotsGbModel import ShuffledBotsGbModel


def execute_single_bot_ordered_training():
    root = Parser.get_project_root()
    path = join(root, 'cached_files', 'cached_datasets', file_name)
    with open(path, 'rb') as input_file:
        data_wizard_datasets: list[GeneralDataset] = load(input_file)

    for bot_dataset in data_wizard_datasets:
        model = SingleBotGbModel(model_name='GBoost')
        print('Training on bot:' + str(bot_dataset.bot_identifier))
        model.update_dataset(new_dataset=bot_dataset)

        model.single_bot_training()

        model.single_bot_evaluation(downsampling=downsampling,
                                    features=feature_names)


def execute_single_bot_shuffled_training():
    root = Parser.get_project_root()
    path = join(root, 'cached_files', 'cached_datasets', file_name)
    with open(path, 'rb') as input_file:
        data_wizard_datasets: list[GeneralDataset] = load(input_file)

    model = ShuffledBotsGbModel(datasets=data_wizard_datasets,
                                model_name='ShuffledGBoost')

    model.train()

    model.compute_test_performance(downsampling=downsampling,
                                   features=feature_names)


if __name__ == "__main__":
    file_name = 'DISP_56exp_ALL_features_50downsampled.pkl'
    feature_names = 'ALL + Global'
    downsampling = Parser.read_down_sampling_size()
    execute_single_bot_shuffled_training()
