import numpy as np


class Polynom:
    def __init__(self):
        self.weights = np.random.normal(size=3)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.weights[0] * x**2 + self.weights[1] * x + self.weights[2]

    def gradent(self, x: np.ndarray) -> np.ndarray:
        return np.vstack([x**2, x, np.ones_like(x)])

    def update_weights(self, delta: np.ndarray) -> None:
        self.weights -= delta
