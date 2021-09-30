import numpy as np
import matplotlib.pyplot as plt
import pickle

from datetime import datetime

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import PrecisionRecallDisplay, ConfusionMatrixDisplay, precision_score, recall_score, f1_score

from src.Utils.Parser import Parser
from src.Utils.data_utils.BotDataset import BotDataset


class GbModel:
    def __init__(self, datasets: list[BotDataset]):
        self.datasets: list[BotDataset] = datasets
        self.model: GradientBoostingClassifier = self.build_model()

    @staticmethod
    def build_model() -> GradientBoostingClassifier:
        seed = Parser.read_seed()
        model = GradientBoostingClassifier(n_estimators=100,
                                           learning_rate=0.1,
                                           max_depth=3,
                                           random_state=seed,
                                           verbose=2)
        return model

    def train(self):

        train_dataset, target_dataset = self.prepare_train_dataset()

        self.model.fit(X=train_dataset, y=target_dataset)

        with open('../cached_files/cached_trained_models/gb_model' + datetime.now().strftime('%d-%m-%Y_%H-%M') + '.pkl', 'wb') as f:
            pickle.dump(self.model, f)

        test_dataset, target_test_dataset = self.prepare_test_dataset()

        test_prediciton = self.model.predict(X=test_dataset)

        score = self.model.score(X=test_dataset, y=target_test_dataset)

        conf_disp = ConfusionMatrixDisplay.from_predictions(y_true=target_test_dataset, y_pred=test_prediciton)
        prec_rec_disp = PrecisionRecallDisplay.from_predictions(y_true=target_test_dataset, y_pred=test_prediciton)
        prec_result = precision_score(y_true=target_test_dataset, y_pred=test_prediciton)
        rec_result = recall_score(y_true=target_test_dataset, y_pred=test_prediciton)
        f1_result = f1_score(y_true=target_test_dataset, y_pred=test_prediciton)

        conf_disp.plot()
        plt.title('Confusion Matrix Gradient Boosting Flocking Down Sampled x20')
        plt.show()

        prec_rec_disp.plot()
        plt.title('Precision Recall Cruve Gradient Boosting Down Sampled x20')
        plt.show()

        print('Mean Accuracy score: ' + str(score))
        print('Precision: ' + str(prec_result))
        print('Recall: ' + str(rec_result))
        print('F1 Score: ' + str(f1_result))

    @staticmethod
    def flatten_dataset(array: np.ndarray):
        return np.reshape(
            array, (array.shape[0], array.shape[1] * array.shape[2])
        )

    @staticmethod
    def training_condition(bot_dataset: BotDataset) -> bool:
        return any(bot_dataset.target_train_dataset)

    def prepare_train_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        merged_train_dataset = np.concatenate(
            [self.flatten_dataset(bot_dataset.train_dataset) for bot_dataset in self.datasets]
        )
        merged_target_train_dataset = np.concatenate(
            [bot_dataset.target_train_dataset for bot_dataset in self.datasets]
        )

        merged_validation_dataset = np.concatenate(
            [self.flatten_dataset(bot_dataset.validation_dataset) for bot_dataset in self.datasets]
        )
        merged_target_validation_dataset = np.concatenate(
            [bot_dataset.target_validation_dataset for bot_dataset in self.datasets]
        )

        train_dataset = np.concatenate([merged_train_dataset, merged_validation_dataset], axis=0)
        target_dataset = np.concatenate([merged_target_train_dataset, merged_target_validation_dataset], axis=0)

        return train_dataset, target_dataset

    def prepare_test_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        merged_test_dataset = np.concatenate(
            [self.flatten_dataset(bot_dataset.test_dataset) for bot_dataset in self.datasets]
        )
        merged_target_test_dataset = np.concatenate(
            [bot_dataset.target_test_dataset for bot_dataset in self.datasets]
        )

        return merged_test_dataset, merged_target_test_dataset

    def saved_train_plot_performances(self):
        with open('../cached_files/cached_trained_models/gb_model30-09-2021_15-39.pkl', 'rb') as f:
            self.model = pickle.load(f)

        test_dataset, target_test_dataset = self.prepare_test_dataset()

        test_prediciton = self.model.predict(X=test_dataset)

        score = self.model.score(X=test_dataset, y=target_test_dataset)

        conf_disp = ConfusionMatrixDisplay.from_predictions(y_true=target_test_dataset, y_pred=test_prediciton)
        prec_rec_disp = PrecisionRecallDisplay.from_predictions(y_true=target_test_dataset, y_pred=test_prediciton)
        prec_result = precision_score(y_true=target_test_dataset, y_pred=test_prediciton)
        rec_result = recall_score(y_true=target_test_dataset, y_pred=test_prediciton)
        f1_result = f1_score(y_true=target_test_dataset, y_pred=test_prediciton)

        conf_disp.plot()
        plt.title('Confusion Matrix Gradient Boosting Flocking Down Sampled x20')
        plt.show()

        prec_rec_disp.plot()
        plt.title('Precision Recall Cruve Gradient Boosting Down Sampled x20')
        plt.show()

        print('Mean Accuracy score: ' + str(score))
        print('Precision: ' + str(prec_result))
        print('Recall: ' + str(rec_result))
        print('F1 Score: ' + str(f1_result))
