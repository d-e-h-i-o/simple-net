import numpy as np


def relu(x):
    return (x > 0) * x


def relu2deriv(output):
    return output > 0


class SimpleNet:
    def __init__(self):
        self.weights_0_1 = np.random.random((784, 264))
        self.weights_1_2 = np.random.random((264, 10))

    def forward(self, x):
        # TODO: add normalization between layers
        layer_1 = relu(x.dot(self.weights_0_1))
        layer_2 = layer_1.dot(self.weights_1_2)
        return layer_2

    def backpropagate(self):
        pass

    def predict(self, x):
        # TODO: change to softmax
        return np.argmax(self.forward(x))
