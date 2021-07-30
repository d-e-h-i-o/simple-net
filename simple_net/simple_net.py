import numpy as np


def relu(x):
    return (x > 0) * x


def relu2deriv(output):
    return output > 0


class SimpleNet:
    def __init__(self, alpha=0.1):
        self.weights_0_1 = np.random.random((784, 264))
        self.weights_1_2 = np.random.random((264, 10))
        self.alpha = alpha

    def forward(self, x):
        # TODO: add normalization between layers
        layer_1 = relu(x.dot(self.weights_0_1))
        layer_2 = layer_1.dot(self.weights_1_2)
        return layer_2, layer_1

    def backpropagate(self, x, y):
        layer_2, layer_1 = self.forward(x)
        goal = np.zeros(10)
        goal[y] = 1
        error = (goal - layer_2) ** 2
        layer_2_delta = goal - layer_2
        layer_1_delta = layer_2_delta.dot(self.weights_1_2.T) * relu2deriv(layer_1)

        self.weights_1_2 += self.alpha * layer_1.T.dot(layer_2_delta)
        self.weights_0_1 += self.alpha * x.T.dot(layer_1_delta)

    def predict(self, x):
        # TODO: change to softmax
        layer_2, _ = self.forward(x)
        return np.argmax(layer_2)
