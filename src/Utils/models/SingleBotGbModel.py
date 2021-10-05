from sklearn.ensemble import GradientBoostingClassifier

from src.Utils.Parser import Parser
from src.Utils.models.SingleBotScikitModel import SingleBotScikitModel


class SingleBotGbModel(SingleBotScikitModel):
    def __init__(self, model_name: str = 'GBoost'):
        seed = Parser.read_seed()
        model = GradientBoostingClassifier(n_estimators=100,
                                           learning_rate=0.1,
                                           max_depth=3,
                                           random_state=seed,
                                           verbose=0)
        super(SingleBotGbModel, self).__init__(model=model,
                                               model_name=model_name)
