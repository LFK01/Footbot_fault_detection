import random
from os.path import join
import matplotlib.pyplot as plt
import pickle

from datetime import datetime

import numpy as np
from sklearn.metrics import PrecisionRecallDisplay, ConfusionMatrixDisplay, precision_score, recall_score, f1_score

from src.Utils.Parser import Parser
from src.Utils.data_utils.datasets.GeneralDataset import GeneralDataset


class ShuffledBotsScikitModel:
    def __init__(self,
                 datasets: list[GeneralDataset],
                 model,
                 model_name: str = 'Generic'):
        random.seed()
        self.datasets = datasets
        self.model_name = model_name
        self.model = model

        self.concatenated_train_datasets = np.concatenate(
            [dataset.train_dataset for dataset in self.datasets]
        )
        self.concatenated_target_train_datasets = np.concatenate(
            [dataset.target_train_dataset for dataset in self.datasets]
        )
        self.concatenated_test_datasets = np.concatenate(
            [dataset.test_dataset for dataset in self.datasets]
        )
        self.concatenated_target_test_datasets = np.concatenate(
            [dataset.target_test_dataset for dataset in self.datasets]
        )

        np.random.seed(Parser.read_seed())
        self.shuffle_datasets()

    def train(self):
        self.model.fit(X=self.concatenated_train_datasets, y=self.concatenated_target_train_datasets)

        with open('../cached_files/cached_trained_models/' + self.model_name + '_model'
                  + datetime.now().strftime('%d-%m-%Y_%H-%M') + '.pkl', 'wb') as f:
            pickle.dump(self.model, f)

    def compute_test_performance(self,
                                 downsampling: int,
                                 features: str):
        test_prediction = self.model.predict(X=self.concatenated_test_datasets)

        score = self.model.score(X=self.concatenated_test_datasets, y=self.concatenated_target_test_datasets)

        ConfusionMatrixDisplay.from_estimator(estimator=self.model,
                                              X=self.concatenated_test_datasets,
                                              y=self.concatenated_target_test_datasets)
        title = 'All bots Shuffled Conf Matrix ' + self.model_name + ' Downsampl x' \
                + str(downsampling) + ' ' \
                + features
        plt.title(title)
        title = title.replace(" + ", "")
        title = title.replace(" ", "_")
        path = join(Parser.get_project_root(), 'images', title)
        plt.savefig(path)

        PrecisionRecallDisplay.from_estimator(estimator=self.model,
                                              X=self.concatenated_test_datasets,
                                              y=self.concatenated_target_test_datasets,
                                              name=self.model_name)

        title = 'All bots shuffled Prec Rec Curve ' + self.model_name + ' Downsampl' \
                + str(downsampling) + ' ' \
                + features
        plt.title(title)
        title = title.replace(" + ", "")
        title = title.replace(" ", "_")
        path = join(Parser.get_project_root(), 'images', title)
        plt.savefig(path)

        prec_result = precision_score(y_true=self.concatenated_target_test_datasets, y_pred=test_prediction)
        rec_result = recall_score(y_true=self.concatenated_target_test_datasets, y_pred=test_prediction)
        f1_result = f1_score(y_true=self.concatenated_target_test_datasets, y_pred=test_prediction)

        print('& {:.4} & {:.4} & {:.4} & {:.4} \\\\'.format(score,
                                                            prec_result,
                                                            rec_result,
                                                            f1_result))
        print('All bots shuffled Mean Accuracy score: ' + str(score))
        print('All bots shuffled Precision: ' + str(prec_result))
        print('All bots shuffled Recall: ' + str(rec_result))
        print('All bots shuffled F1 Score: ' + str(f1_result))

    def shuffle_datasets(self):
        assert len(self.concatenated_train_datasets) == len(self.concatenated_target_train_datasets)
        permutation = np.random.permutation(len(self.concatenated_train_datasets))
        self.concatenated_train_datasets = self.concatenated_train_datasets[permutation]
        self.concatenated_target_train_datasets = self.concatenated_target_train_datasets[permutation]

        assert len(self.concatenated_test_datasets) == len(self.concatenated_test_datasets)
        permutation = np.random.permutation(len(self.concatenated_test_datasets))
        self.concatenated_test_datasets = self.concatenated_test_datasets[permutation]
        self.concatenated_target_test_datasets = self.concatenated_target_test_datasets[permutation]
