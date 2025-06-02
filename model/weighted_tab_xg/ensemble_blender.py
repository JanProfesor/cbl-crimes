import numpy as np
from sklearn.metrics import mean_squared_error

class EnsembleBlender:
    @staticmethod
    def find_best_weight(pred1, pred2, y_true):
        weights = np.linspace(0, 1, 101)
        rmses = [np.sqrt(mean_squared_error(y_true, w*pred1 + (1-w)*pred2)) for w in weights]
        best_w = weights[np.argmin(rmses)]
        return best_w, min(rmses)