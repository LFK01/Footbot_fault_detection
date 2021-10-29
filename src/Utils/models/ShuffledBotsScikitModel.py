import random
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        self.shuffle_datasets()

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
        path = join(Parser.return_performance_image_directory_path(task_name), title)
        plt.savefig(path)


    def shuffle_datasets(self):
        assert len(self.datasets.train_dataset) == len(self.datasets.target_train_dataset)
        permutation = np.random.permutation(len(self.datasets.train_dataset))
        self.datasets.train_dataset = self.datasets.train_dataset[permutation]
        self.datasets.target_train_dataset = self.datasets.target_train_dataset[permutation]

        assert len(self.datasets.test_dataset) == len(self.datasets.target_test_dataset)
        permutation = np.random.permutation(len(self.datasets.test_dataset))
        self.datasets.test_dataset = self.datasets.test_dataset[permutation]
        self.datasets.target_test_dataset = self.datasets.target_test_dataset[permutation]
