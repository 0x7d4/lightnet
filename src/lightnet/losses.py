import numpy as np


class Loss:
    """Base class for loss functions"""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculates loss"""
        return self.get(y_true, y_pred)

    def get(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculates loss"""
        raise NotImplementedError

    def d(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Computes derivative of loss function"""
        raise NotImplementedError


class MSE(Loss):
    """Mean Squared Error loss function"""

    def get(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return (y_true - y_pred) ** 2

    def d(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        assert y_true.size != 0, "y_true is empty"
        return -2 * (y_true - y_pred)


class BinaryCrossEntropy(Loss):
    """Binary Cross Entropy loss function"""

    def get(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def d(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        assert y_true.size != 0, "y_true is empty"
        return y_pred - y_true
