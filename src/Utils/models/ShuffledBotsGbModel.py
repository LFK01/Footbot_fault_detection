from sklearn.ensemble import GradientBoostingClassifier

from src.Utils.Parser import Parser
from src.Utils.data_utils.datasets_classes.GeneralDataset import GeneralDataset
from src.Utils.models.ShuffledBotsScikitModel import ShuffledBotsScikitModel


class ShuffledBotsGbModel(ShuffledBotsScikitModel):
    def __init__(self,
                 datasets: list[GeneralDataset],
                 model_name: str = 'ShuffledGBoost'):
        seed = Parser.read_seed()
        model = GradientBoostingClassifier(loss='exponential',
                                           n_estimators=100,
                                           learning_rate=0.1,
                                           max_depth=5,
                                           random_state=seed,
                                           verbose=2)
        super(ShuffledBotsGbModel, self).__init__(model=model,
                                                  datasets=datasets,
                                                  model_name=model_name)
