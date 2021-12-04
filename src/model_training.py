from os import listdir
from os.path import join
from pickle import load
from typing import List, Dict
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
                                         feature_set_name: str,
                                         filename_date: str,
                                         feature_set_delta_time: float):
    with open(file_name, 'rb') as input_file:
        data_wizard_datasets: GeneralDataset = load(input_file)

    model = ShuffledBotsGbModel(datasets=data_wizard_datasets,
                                model_name='ShuffledGBoost')

    train_start_time = datetime.now()
    model.train()
    train_end_time = datetime.now()
    delta_time = train_end_time - train_start_time

    model.compute_test_performance_default_model(task_name=task_name,
                                                 feature_set_name=feature_set_name,
                                                 filename_date=filename_date,
                                                 training_time=delta_time.total_seconds(),
                                                 feature_set_building_time=feature_set_delta_time)


def execute_training_feature_set_datasets(task_name: str,
                                          delta_times_dict: Dict[str, float]):
    cached_dataset_directory_path = Parser.return_cached_dataset_directory_path(task_name)

    filename_date = datetime.now().strftime('%d-%m-%Y_%H-%M')
    for file in listdir(cached_dataset_directory_path):
        print('Training on: {}'.format(file))
        path = join(cached_dataset_directory_path, file)
        feature_set_name_from_file = file.split('_')[3:-2]
        feature_set_name_from_file = '_'.join(feature_set_name_from_file)
        execute_single_bot_shuffled_training(task_name=task_name,
                                             file_name=path,
                                             feature_set_name=feature_set_name_from_file,
                                             filename_date=filename_date,
                                             feature_set_delta_time=delta_times_dict[feature_set_name_from_file])


if __name__ == "__main__":
    pass
