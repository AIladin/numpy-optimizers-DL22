import numpy as np

from .sgd import SGD


class Momentum(SGD):
    def __init__(
        self,
        *args,
    ):
        super().__init__(*args)
        self.momentum: float = 0

    def step(self, x: np.ndarray, y: np.ndarray) -> float:
        predicted = self.optimizable(x)
        loss = self.loss_fn(y, predicted)
        grad = self.gradient(x, y, predicted)
        # todo calculate and apply momentum
        self.optimizable.update_weights(grad * self.learning_rate)
        return loss

