from src.layers.dense import Layer_Dense
from src.activations.relu import Activation_ReLU
from src.activations.softmax import Activation_Softmax
from src.losses.softmax_categorical_crossentropy import Activation_Softmax_Loss_CategoricalCrossentropy
from src.losses.categorical_crossentropy import Loss_CategoricalCrossentropy
from src.metrics.accuracy import Accuracy

import matplotlib as plt

class NeuralNetwork:
    def __init__(self):
        self.dense1 = Layer_Dense(2, 32)
        self.activation1 = Activation_ReLU()
        self.dense2 = Layer_Dense(32, 64)
        self.activation2 = Activation_ReLU()
        self.dense3 = Layer_Dense(64, 32)
        self.activation3 = Activation_ReLU()
        self.dense4 = Layer_Dense(32, 3)
        self.loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

    def forward(self, X, y):
        self.dense1.forward(X)
        self.activation1.forward(self.dense1.output)
        self.dense2.forward(self.activation1.output)
        self.activation2.forward(self.dense2.output)
        self.dense3.forward(self.activation2.output)
        self.activation3.forward(self.dense3.output)
        self.dense4.forward(self.activation3.output)
        loss = self.loss_activation.forward(self.dense4.output, y)
        return loss

    def backward(self, y):
        self.loss_activation.backward(self.loss_activation.output, y)
        self.dense4.backward(self.loss_activation.dinputs)
        self.activation3.backward(self.dense4.dinputs)
        self.dense3.backward(self.activation3.dinputs)
        self.activation2.backward(self.dense3.dinputs)
        self.dense2.backward(self.activation2.dinputs)
        self.activation1.backward(self.dense2.dinputs)
        self.dense1.backward(self.activation1.dinputs)




