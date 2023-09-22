import numpy as np
from typing import List
from .layers import Dense
from .losses import Loss
from prettytable import PrettyTable


class Model:
    """Base module for NN models"""

    def __init__(self, loss_fn: Loss, layers: List[Dense]) -> None:
        self.loss_fn = loss_fn
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    predict = forward

    def backward(self) -> np.ndarray:
        raise NotImplementedError

    def train(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        epochs: int = 100,
        lr: float = 1e-4,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> None:
        raise NotImplementedError

    def get_info(self) -> str:
        t = PrettyTable()
        t.title = f"{FullyConnected.__name__} P:{self.params_n_humanized}"
        t.field_names = ["Layer", "Shape", "Activation"]
        for layer in self.layers:
            t.add_row(
                [
                    layer.id,
                    f"({layer.W.shape[0]}, {layer.W.shape[1] if layer.id > 1 else 'X'})",
                    layer.activation.__class__.__name__ if layer.activation else "None",
                ]
            )
        return t.get_string()

    @property
    def params_n(self):
        """Returns the total number of model parameters"""
        n = 0
        for layer in self.layers:
            n += layer.W.size
        return n

    @property
    def params_n_humanized(self):
        """Returns the total number of model parameters in a human readable format"""
        # using k, m, b
        n = self.params_n
        if n < 1e3:
            s = str(n)
        elif n < 1e6:
            s = f"{str(n)[:-3]}k"
        elif n < 1e9:
            s = f"{str(n)[:-6]}m"
        return s

    def save(self, path):
        """Saves the model's parameters to a file. Uses
        `numpy.savez_compressed` to save the parameters."""
        data = {}
        for layer in self.layers:
            data[f"W{layer.id}"] = layer.W
            data[f"b{layer.id}"] = layer.b
        np.savez_compressed(path, **data)
        print(f"Model saved to {path}")

    def load(self, path):
        """Loads the params from a file."""
        data = np.load(path)
        for layer in self.layers:
            layer.W = data[f"W{layer.id}"]
            layer.b = data[f"b{layer.id}"]
        print(f"Model loaded from {path}")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def __str__(self) -> str:
        return self.get_info()


class FullyConnected(Model):
    def __init__(self, loss_fn, layers: List[Dense]) -> None:
        super().__init__(loss_fn, layers)
        for i in range(1, len(layers)):
            self.layers[i].attach(layers[i - 1])

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        a = x
        for layer in self.layers:
            a = layer(a)
        return a

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray, lr: float):
        dA_prev = self.loss_fn.d(y_true, y_pred)
        for layer in self.layers[::-1]:
            layer.dA = dA_prev
            layer.dZ = layer.dA * layer.activation.d(layer.Z, cache=layer.A)
            A_prev = layer.prev.A if layer.prev else self.x
            layer.dW = np.dot(layer.dZ, A_prev.T)
            layer.db = np.sum(layer.dZ, axis=1, keepdims=True)
            dA_prev = np.dot(layer.W.T, layer.dZ)
            layer.update(lr)

    def train_step(self, x: np.ndarray, y: np.ndarray, lr: float):
        y_pred = self.forward(x)
        self.backward(y, y_pred, lr)
        loss = self.loss_fn(y, y_pred)
        return loss

    def train(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 1,
        verbose: bool = True,
    ):
        self.layers[0].init_params((self.layers[0].N, X.shape[0]), batch_size)
        self.m = X.shape[1]
        loss_per_instance = []
        losses = []
        print("")
        for i in range(epochs):
            loss_per_instance = []
            for j in range(self.m):
                x = X[:, j].reshape(-1, 1)
                y = Y[:, j].reshape(-1, 1)
                loss_per_instance.append(self.train_step(x, y, lr))
            losses.append(np.mean(loss_per_instance))
            if verbose and (i % 1 == 0 or i == epochs - 1):
                l = np.round(losses[-1], 4)
                print(f"Epoch {i}: {l}")
        print("")
        return losses
