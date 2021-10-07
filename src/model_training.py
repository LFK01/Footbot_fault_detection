from pickle import load

from src.Utils.data_utils.datasets.GeneralDataset import GeneralDataset
from src.Utils.models.SingleBotGbModel import SingleBotGbModel
from src.Utils.models.ShuffledBotsGbModel import ShuffledBotsGbModel


def execute_single_bot_ordered_training():
    with open('../cached_files/cached_datasets/' + file_name, 'rb') as input_file:
        data_wizard_datasets: list[GeneralDataset] = load(input_file)

    for bot_dataset in data_wizard_datasets:
        model = SingleBotGbModel(model_name='GBoost')
        print('Training on bot:' + str(bot_dataset.bot_identifier))
        model.update_dataset(new_dataset=bot_dataset)

        model.single_bot_training()

        model.single_bot_evaluation(downsampling=downsampling,
                                    features=feature_names)


def execute_single_bot_shuffled_training():
    with open('../cached_files/cached_datasets/' + file_name, 'rb') as input_file:
        data_wizard_datasets: list[GeneralDataset] = load(input_file)

    model = ShuffledBotsGbModel(datasets=data_wizard_datasets,
                                model_name='ShuffledGBoost')

    model.train()

    model.compute_test_performance(downsampling=downsampling,
                                   features=feature_names)


file_name = '149exp_15bot_datasets_down_sampled_20_ALL_features.pkl'
downsampling = 20
feature_names = 'ALL + Global'

if __name__ == "__main__":
    execute_single_bot_shuffled_training()
