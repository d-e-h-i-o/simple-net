import numpy as np
import sys


def relu(x):
    return (x > 0) * x


def relu2deriv(output):
    return output > 0


# np.set_printoptions(threshold=sys.maxsize)


class SimpleNet:
    def __init__(self, alpha=0.005, hidden_layer_size=264, use_dropout=False):
        self.weights_0_1 = 0.2 * np.random.random((784, hidden_layer_size)) - 0.1
        self.weights_1_2 = 0.2 * np.random.random((hidden_layer_size, 10)) - 0.1
        self.alpha = alpha
        self.dropout_mask = (
            np.random.randint(2, size=(1, hidden_layer_size)) if use_dropout else None
        )

    def forward(self, x):
        x = x.reshape(-1, 784)  # TODO: understand why this is necessary
        # TODO: add normalization between layers
        layer_1 = relu(x.dot(self.weights_0_1))

        if self.dropout_mask is not None:
            layer_1 *= self.dropout_mask * 2

        layer_2 = layer_1.dot(self.weights_1_2)
        return layer_2, layer_1

    def backpropagate(self, x, y):
        x = x.reshape(-1, 784)  # TODO: understand why this is necessary
        layer_2, layer_1 = self.forward(x)
        goal = np.zeros(10)
        goal[y] = 1
        error = (goal - layer_2) ** 2
        layer_2_delta = goal - layer_2
        layer_1_delta = layer_2_delta.dot(self.weights_1_2.T) * relu2deriv(layer_1)

        if self.dropout_mask is not None:
            layer_1_delta *= self.dropout_mask

        self.weights_1_2 += self.alpha * layer_1.T.dot(layer_2_delta)
        self.weights_0_1 += self.alpha * x.T.dot(layer_1_delta)

        return np.sum(error)

    def predict(self, x):
        # TODO: change to softmax
        layer_2, _ = self.forward(x.reshape(-1, 784))
        return np.argmax(layer_2)

    def test_one(self, x, y):
        layer_2, _ = self.forward(x.reshape(-1, 784))
        correct = self.predict(x) == y
        goal = np.zeros(10)
        goal[y] = 1
        error = np.sum((goal - layer_2) ** 2)
        return error, correct

    def train(self, x_train, y_train, epochs=20):

        assert len(x_train) == len(y_train)

        for epoch in range(epochs):
            error = 0
            for i in range(len(x_train)):
                error += self.backpropagate(x_train[i], y_train[i])

            print(f"Error for epoch {epoch}: {error / len(x_train)}")

    def test(self, x_test, y_test):

        assert len(x_test) == len(y_test)

        total_correct, total_error = (0, 0)

        for i in range(len(x_test)):
            error, correct = self.test_one(x_test[i], y_test[i])
            total_correct += correct
            total_error += error

        return total_error / len(x_test), total_correct / len(x_test)
