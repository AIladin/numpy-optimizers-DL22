import numpy as np


class MSELoss:
    def __call__(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        return ((y - y_pred) ** 2).mean()

    def gradient(self, y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return -2 * (y - y_pred) / len(y)
