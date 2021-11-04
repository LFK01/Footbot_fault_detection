from os import listdir
from os.path import join
from pickle import load
from typing import List
from datetime import datetime

from src.Utils.Parser import Parser
from src.Utils.data_utils.datasets_classes.GeneralDataset import GeneralDataset
from src.Utils.models.SingleBotGbModel import SingleBotGbModel
from src.Utils.models.ShuffledBotsGbModel import ShuffledBotsGbModel


def execute_single_bot_ordered_training(file_name: str,
                                        feature_names_list: str,
                                        downsampling: int):
    with open(file_name, 'rb') as input_file:
        data_wizard_datasets: List[GeneralDataset] = load(input_file)

    for bot_dataset in data_wizard_datasets:
        model = SingleBotGbModel(model_name='GBoost')
        print('Training on bot:' + str(bot_dataset.bot_identifier))
        model.update_dataset(new_dataset=bot_dataset)

        model.single_bot_training()

        model.single_bot_evaluation(downsampling=downsampling,
                                    features=feature_names_list)


def execute_single_bot_shuffled_training(task_name: str,
                                         file_name: str,
                                         feature_names_list: str,
                                         downsampling: int,
                                         filename_date: str):
    with open(file_name, 'rb') as input_file:
        data_wizard_datasets: GeneralDataset = load(input_file)

    model = ShuffledBotsGbModel(datasets=data_wizard_datasets,
                                model_name='ShuffledGBoost')

    model.train()

    model.compute_test_performance_default_model(task_name=task_name,
                                                 downsampling=downsampling,
                                                 features=feature_names_list,
                                                 filename_date=filename_date)


def execute_training_feature_set_datasets(task_name: str):
    cached_dataset_directory_path = Parser.return_cached_dataset_directory_path(task_name)
    parser_downsampling = Parser.read_timeseries_down_sampling()

    filename_date = datetime.now().strftime('%d-%m-%Y_%H-%M')
    graph_feature_names = ''
    for file in listdir(cached_dataset_directory_path):
        print('Training on: {}'.format(file))
        if 'set1' in file:
            graph_feature_names = 'set1'
        elif 'set2' in file:
            graph_feature_names = 'set2'
        elif 'set3' in file:
            graph_feature_names = 'set3'
        path = join(cached_dataset_directory_path, file)
        execute_single_bot_shuffled_training(task_name=task_name,
                                             file_name=path,
                                             feature_names_list=graph_feature_names,
                                             downsampling=parser_downsampling,
                                             filename_date=filename_date)


if __name__ == "__main__":
    main_task_name = 'FLOC'
    execute_training_feature_set_datasets(task_name=main_task_name)
