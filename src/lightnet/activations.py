import numpy as np


class Activation:
    """Base class for activation functions"""

    def __call__(self, input: np.ndarray) -> np.ndarray:
        """Applies activation function to input"""
        return self.get(input)

    def get(self, input: np.ndarray) -> np.ndarray:
        """Applies activation function to input"""
        raise NotImplementedError

    def d(self, input: np.ndarray, **kwargs) -> np.ndarray:
        """Computes derivative of activation function"""
        raise NotImplementedError


class Linear(Activation):
    """Linear activation function"""

    def get(self, input: np.ndarray) -> np.ndarray:
        return input

    def d(self, input: np.ndarray, **kwargs) -> np.ndarray:
        return np.ones(input.shape)


class Sigmoid(Activation):
    """Sigmoid activation function"""

    def get(self, input: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-input))

    def d(self, input: np.ndarray, **kwargs) -> np.ndarray:
        sigmoid = kwargs.get("cache")
        if sigmoid is None:
            sigmoid = self.get(input)
        return sigmoid * (1 - sigmoid)


class ReLU(Activation):
    """ReLU activation function"""

    def get(self, input: np.ndarray) -> np.ndarray:
        return np.maximum(0, input)

    def d(self, input: np.ndarray, **kwargs) -> np.ndarray:
        return np.where(input > 0, 1, 0)
