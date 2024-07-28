import numpy as np
from src.losses.loss import Loss

class Loss_MeanSquaredError(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        squared_error = np.square(np.subtract(y_pred,y_true))
        return squared_error

    def backward(self, y_pred, y_true):
        samples = len(y_true)
        outputs = y_pred.shape[-1]
        self.dinputs = -2 * (y_true - y_pred) / outputs
        self.dinputs = self.dinputs / samples