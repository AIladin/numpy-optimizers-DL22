from typing import Protocol

import numpy as np
from tqdm.auto import tqdm, trange


class Loss(Protocol):
    def __call__(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        ...

    def gradient(self, y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        ...


class Optimizable(Protocol):
    weights: np.ndarray

    def __init__(self):
        ...

    def __call__(self, x: np.ndarray) -> np.ndarray:
        ...

    def gradent(self, x: np.ndarray) -> np.ndarray:
        ...

    def update_weights(self, delta: np.ndarray) -> None:
        ...


class SGD:
    def __init__(
        self,
        optimizable: Optimizable,
        loss_fn: Loss,
        learning_rate: float,
    ):

        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.optimizable = optimizable

    def gradient(
        self,
        x: np.ndarray,
        y: np.ndarray,
        predicted: np.ndarray,
    ) -> np.ndarray:
        return self.optimizable.gradent(x) @ self.loss_fn.gradient(y, predicted)

    def step(self, x: np.ndarray, y: np.ndarray) -> float:
        predicted = self.optimizable(x)
        loss = self.loss_fn(y, predicted)
        grad = self.gradient(x, y, predicted)
        self.optimizable.update_weights(grad * self.learning_rate)
        return loss

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int,
        batch_size: int,
        keep_epochs_pbar: bool = True,
        keep_total_pbar: bool = True,
        epoch_verbose: bool = False,
        total_verbose: bool = True,
    ) -> list[float]:
        epoch_history = []
        n_batches = int(np.ceil(len(x) / batch_size))

        epoch_iterator = trange(
            0,
            epochs,
            desc="Model",
            leave=keep_total_pbar,
            disable=not total_verbose,
        )

        for i in epoch_iterator:
            batch_history: list[float] = []
            batch_iterator = tqdm(
                zip(
                    np.array_split(x, n_batches),
                    np.array_split(y, n_batches),
                ),
                desc=f"Epoch {i}",
                total=n_batches,
                leave=keep_epochs_pbar,
                disable=not epoch_verbose,
            )
            for x_batch, y_batch in batch_iterator:
                batch_loss = self.step(x_batch, y_batch)
                batch_history.append(batch_loss)
                batch_iterator.set_postfix({"loss": batch_loss})
            epoch_loss = np.mean(batch_history)
            epoch_history.append(epoch_loss)
            epoch_iterator.set_postfix({"loss": epoch_loss})
        return epoch_history
