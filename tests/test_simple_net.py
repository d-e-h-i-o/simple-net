import pytest
import numpy as np

from simple_net import __version__
from simple_net import SimpleNet, load_mnist
from simple_net.simple_net import relu


def test_version():
    assert __version__ == "0.1.0"


@pytest.fixture
def x():
    ((x_train, y_train), (x_test, y_test), _) = load_mnist()
    return x_test[0]


@pytest.fixture
def y():
    ((x_train, y_train), (x_test, y_test), _) = load_mnist()
    return y_test[0]


def test_simple_net(x, y):
    nn = SimpleNet()
    assert nn.weights_0_1.shape == (784, 264)
    assert nn.weights_1_2.shape == (264, 10)

    assert nn.forward(x)[0].all()
    assert 0 <= nn.predict(x) < 10
    nn.backpropagate(x, y)


def test_relu():
    array = np.array([1, 2, 3, -1, 0])
    expected = np.array([1, 2, 3, 0, 0])

    assert (relu(array) == expected).all()


def test_mnist():
    ((x_train, y_train), (x_test, y_test), _) = load_mnist()
    assert x_train is not None
    assert y_train is not None
    assert x_test is not None
    assert y_test is not None
