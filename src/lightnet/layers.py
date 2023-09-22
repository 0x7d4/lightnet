import numpy as np
from typing import Optional, Tuple, Union
from .activations import Activation, Linear


class Dense:
    def __init__(
        self,
        N: int,
        activation: Activation = Linear(),
        weight_variance: Optional[int] = 0.01,
    ):
        """
        ### Params
        `N` - The number of units used in the layer
        `activation` - The activation function to use
        `weight_variance` - The variance of the weights
        """
        self.N = N
        self.activation = activation
        self.weight_variance = weight_variance
        self.prev = None
        self.init_params((N, 1))

    def attach(self, layer: "Dense"):
        """Connects the previous layer to this layer and
        initilizes the parameters of this layer."""
        self.init_params((self.N, layer.N))
        self.prev = layer

    def update(self, lr: float):
        """Updates the parameters with new given gradients"""
        self.W = self.W - (lr * self.dW)
        self.b = self.b - (lr * self.db)

    def predict(self, X: np.ndarray):
        self.Z = np.dot(self.W, X) + self.b
        self.A = self.activation(self.Z)
        return self.A

    def init_params(self, w_shape: Tuple[int, int], batch_size: int = 1):
        self.W = np.random.randn(*w_shape) * self.weight_variance
        self.Z = np.zeros((self.W.shape[0], batch_size))
        self.A = np.zeros(self.Z.shape)
        self.b = np.zeros((self.W.shape[0], 1))
        #
        self.dW = np.zeros(self.W.shape)
        self.dZ = np.zeros(self.Z.shape)
        self.dA = np.zeros(self.A.shape)
        self.db = np.zeros(self.b.shape)

    @property
    def id(self):
        return self.prev.id + 1 if self.prev else 1

    def __call__(self, X: np.ndarray):
        """
        ### Params
        `x` - Input
        """
        return self.predict(X)
