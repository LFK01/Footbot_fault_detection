import random
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir, mkdir
from os.path import join
from datetime import datetime

from sklearn.metrics import PrecisionRecallDisplay, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from sklearn.inspection import permutation_importance

from src.Utils.Parser import Parser
from src.Utils.data_utils.datasets_classes.GeneralDataset import GeneralDataset


class ShuffledBotsScikitModel:
    def __init__(self,
                 datasets: GeneralDataset,
                 model,
                 model_name: str = 'Generic'):
        random.seed()
        self.datasets = datasets
        self.model_name = model_name
        self.model = model

        np.random.seed(Parser.read_seed())

    def train(self):
        self.model.fit(X=self.datasets.train_dataset, y=self.datasets.target_train_dataset)

        root = Parser.get_project_root()
        model_string = self.model_name + '_model' + datetime.now().strftime('%d-%m-%Y_%H-%M') + '.pkl'
        path = join(root, 'cached_files', 'cached_trained_models', model_string)

        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def compute_test_performance_default_model(self,
                                               task_name: str,
                                               downsampling: int,
                                               features: str,
                                               filename_date: str):
        test_prediction = self.model.predict(X=self.datasets.test_dataset)

        score = self.model.score(X=self.datasets.test_dataset,
                                 y=self.datasets.target_test_dataset)

        ConfusionMatrixDisplay.from_estimator(estimator=self.model,
                                              X=self.datasets.test_dataset,
                                              y=self.datasets.target_test_dataset)
        title = 'All bots Shuffled Conf Matrix ' + self.model_name + ' Downsampl x' \
                + str(downsampling) + ' ' \
                + features
        plt.title(title)
        title = title.replace(" + ", "")
        title = title.replace(" ", "_")
        saving_images_path = join(Parser.return_performance_image_directory_path(task_name), title)
        plt.savefig(saving_images_path)

        PrecisionRecallDisplay.from_estimator(estimator=self.model,
                                              X=self.datasets.test_dataset,
                                              y=self.datasets.target_test_dataset,
                                              name=self.model_name)

        title = 'All bots shuffled Prec Rec Curve ' + self.model_name + ' Downsampl' \
                + str(downsampling) + ' ' \
                + features
        plt.title(title)
        title = title.replace(" + ", "")
        title = title.replace(" ", "_")
        saving_images_path = join(Parser.return_performance_image_directory_path(task_name), title)
        plt.savefig(saving_images_path)

        prec_result = precision_score(y_true=self.datasets.target_test_dataset, y_pred=test_prediction)
        rec_result = recall_score(y_true=self.datasets.target_test_dataset, y_pred=test_prediction)
        f1_result = f1_score(y_true=self.datasets.target_test_dataset, y_pred=test_prediction)

        output_log = 'Model: {} DownSampling: {} FeatureSet: {} \n'.format(self.model_name, downsampling, features)
        output_log += '& {:.4} & {:.4} & {:.4} & {:.4} \\\\ \n'.format(score, prec_result, rec_result, f1_result)
        output_log += 'All bots shuffled Mean Accuracy score: {} \n'.format(score)
        output_log += 'All bots shuffled Precision: {} \n'.format(prec_result)
        output_log += 'All bots shuffled Recall: {} \n'.format(rec_result)
        output_log += 'All bots shuffled F1 Score: {} \n'.format(f1_result)
        print(output_log)

        training_log_files_path = join(Parser.get_project_root(), 'txt_files')
        if task_name + '_training_logs' not in listdir(training_log_files_path):
            mkdir(join(training_log_files_path, task_name + '_training_logs'))
        training_log_files_path = join(training_log_files_path, task_name + '_training_logs')

        file_path = join(training_log_files_path, 'training_log' + filename_date + '.txt')
        with open(file_path, 'a+') as output_file:
            output_file.write(output_log)

        feature_importance = permutation_importance(self.model,
                                                    X=self.datasets.test_dataset,
                                                    y=self.datasets.target_test_dataset,
                                                    random_state=Parser.read_seed())
        model_importances = pd.Series(feature_importance.importances_mean,
                                      index=self.datasets.feature_names)

        fig, ax = plt.subplots()
        model_importances.plot.bar(yerr=feature_importance.importances_std, ax=ax)
        title = 'Feature importances using permutation n_feats_' + str(len(self.datasets.feature_names))
        ax.set_title(title)
        ax.set_ylabel('Mean accuracy decrease')
        fig.tight_layout()
        title = title.replace(' + ', '')
        title = title.replace(' ', '_')
        saving_images_path = join(Parser.return_performance_image_directory_path(task_name), title)
        plt.savefig(saving_images_path)
