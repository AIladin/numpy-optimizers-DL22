import numpy as np

from .sgd import SGD


class Momentum(SGD):
    def __init__(self, *args, momentum: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum: float = momentum
        self.change: np.ndarray = np.zeros_like(self.optimizable.weights)

    def step(self, x: np.ndarray, y: np.ndarray) -> float:
        predicted = self.optimizable(x)
        loss = self.loss_fn(y, predicted)
        grad = self.gradient(x, y, predicted)
        self.change = self.learning_rate * grad + self.momentum * self.change
        self.optimizable.update_weights(self.change)
        return loss
