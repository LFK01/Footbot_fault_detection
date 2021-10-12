import matplotlib.pyplot as plt
import pickle
from os.path import join

from datetime import datetime

from sklearn.metrics import PrecisionRecallDisplay, ConfusionMatrixDisplay, precision_score, recall_score, f1_score

from src.Utils.data_utils.datasets_classes.GeneralDataset import GeneralDataset
from src.Utils.Parser import Parser


class SingleBotScikitModel:
    def __init__(self,
                 model,
                 model_name: str = 'Generic'):
        self.datasets = None
        self.model_name = model_name
        self.model = model

    def update_dataset(self,
                       new_dataset: GeneralDataset):
        self.datasets = new_dataset

    def single_bot_training(self):
        self.model.fit(X=self.datasets.train_dataset, y=self.datasets.target_train_dataset)

    def save_trained_model(self):
        with open('../cached_files/cached_trained_models/' + self.model_name + '_model'
                  + datetime.now().strftime('%d-%m-%Y_%H-%M') + '.pkl', 'wb') as f:
            pickle.dump(self.model, f)

    def single_bot_evaluation(self,
                              downsampling: int,
                              features: str):
        self.compute_test_performance(downsampling=downsampling,
                                      features=features)

    def compute_test_performance(self,
                                 downsampling: int,
                                 features: str):
        test_prediction = self.model.predict(X=self.datasets.test_dataset)

        score = self.model.score(X=self.datasets.test_dataset, y=self.datasets.target_test_dataset)

        ConfusionMatrixDisplay.from_estimator(estimator=self.model,
                                              X=self.datasets.test_dataset,
                                              y=self.datasets.target_test_dataset)
        title = 'Bot' + str(self.datasets.bot_identifier) + ' Conf Matrix ' + self.model_name + ' Downsampl x' \
                + str(downsampling) + ' ' \
                + features
        plt.title(title)
        title = title.replace("+", "")
        title = title.replace(" ", "_")

        path = join(Parser.get_project_root(), 'images', title)
        plt.savefig(path)

        PrecisionRecallDisplay.from_estimator(estimator=self.model,
                                              X=self.datasets.test_dataset,
                                              y=self.datasets.target_test_dataset,
                                              name=self.model_name)
        title = 'Bot' + str(self.datasets.bot_identifier) + ' Prec Rec Curve ' + self.model_name + ' Downsampl' \
                + str(downsampling) + ' ' \
                + features
        plt.title(title)
        title = title.replace(" + ", "")
        title = title.replace(" ", "_")
        path = join(Parser.get_project_root(), 'images', title)
        plt.savefig(path)

        prec_result = precision_score(y_true=self.datasets.target_test_dataset, y_pred=test_prediction)
        rec_result = recall_score(y_true=self.datasets.target_test_dataset, y_pred=test_prediction)
        f1_result = f1_score(y_true=self.datasets.target_test_dataset, y_pred=test_prediction)

        print('{} & {:.4} & {:.4} & {:.4} & {:.4} \\\\'.format(self.datasets.bot_identifier,
                                                               score,
                                                               prec_result,
                                                               rec_result,
                                                               f1_result))
        # print('Bot' + str(self.datasets_classes.bot_identifier) + ' Mean Accuracy score: ' + str(score))
        # print('Bot' + str(self.datasets_classes.bot_identifier) + ' Precision: ' + str(prec_result))
        # print('Bot' + str(self.datasets_classes.bot_identifier) + ' Recall: ' + str(rec_result))
        # print('Bot' + str(self.datasets_classes.bot_identifier) + ' F1 Score: ' + str(f1_result))
