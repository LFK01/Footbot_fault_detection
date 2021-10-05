from sklearn.neural_network import MLPClassifier

from src.Utils.Parser import Parser
from src.Utils.models.SingleBotScikitModel import SingleBotScikitModel


class SingleBotNNModel(SingleBotScikitModel):
    def __init__(self, model_name: str = 'FFNN'):
        seed = Parser.read_seed()
        model = MLPClassifier(hidden_layer_sizes=(256, 256, 128, 128, 64, 64, 64),
                              solver='adam',
                              alpha=1e-5,
                              learning_rate='invscaling',
                              verbose=True,
                              early_stopping=True,
                              n_iter_no_change=25,
                              random_state=seed)
        super(SingleBotNNModel, self).__init__(model=model,
                                               model_name=model_name)

