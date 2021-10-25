import random
from os.path import join
import matplotlib.pyplot as plt
import pickle

from datetime import datetime

import numpy as np
from sklearn.metrics import PrecisionRecallDisplay, ConfusionMatrixDisplay, precision_score, recall_score, f1_score

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
        self.shuffle_datasets()

    def train(self):
        self.model.fit(X=self.datasets.train_dataset, y=self.datasets.target_train_dataset)

        root = Parser.get_project_root()
        model_string = self.model_name + '_model' + datetime.now().strftime('%d-%m-%Y_%H-%M') + '.pkl'
        path = join(root, 'cached_files', 'cached_trained_models', model_string)

        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def compute_test_performance(self,
                                 task_name: str,
                                 downsampling: int,
                                 features: str):
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
        path = join(Parser.return_performance_image_directory_path(task_name), title)
        plt.savefig(path)

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
        path = join(Parser.return_performance_image_directory_path(task_name), title)
        plt.savefig(path)

        prec_result = precision_score(y_true=self.datasets.target_test_dataset, y_pred=test_prediction)
        rec_result = recall_score(y_true=self.datasets.target_test_dataset, y_pred=test_prediction)
        f1_result = f1_score(y_true=self.datasets.target_test_dataset, y_pred=test_prediction)

        print('& {:.4} & {:.4} & {:.4} & {:.4} \\\\'.format(score,
                                                            prec_result,
                                                            rec_result,
                                                            f1_result))
        print('All bots shuffled Mean Accuracy score: ' + str(score))
        print('All bots shuffled Precision: ' + str(prec_result))
        print('All bots shuffled Recall: ' + str(rec_result))
        print('All bots shuffled F1 Score: ' + str(f1_result))

    def shuffle_datasets(self):
        assert len(self.datasets.train_dataset) == len(self.datasets.target_train_dataset)
        permutation = np.random.permutation(len(self.datasets.train_dataset))
        self.datasets.train_dataset = self.datasets.train_dataset[permutation]
        self.datasets.target_train_dataset = self.datasets.target_train_dataset[permutation]

        assert len(self.datasets.test_dataset) == len(self.datasets.target_test_dataset)
        permutation = np.random.permutation(len(self.datasets.test_dataset))
        self.datasets.test_dataset = self.datasets.test_dataset[permutation]
        self.datasets.target_test_dataset = self.datasets.target_test_dataset[permutation]
