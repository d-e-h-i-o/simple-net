from simple_net import SimpleNet, load_mnist

def benchmark():
    ((x_train, y_train), (x_test, y_test), _) = load_mnist()

    nn = SimpleNet(alpha=0.0001)
    accuracy = nn.test(x_test[:1000], y_test[:1000])
    print(f"Untrained accuracy: {accuracy}")
    print(nn.weights_1_2)

    nn.train(x_train, y_train)
    accuracy_trained = nn.test(x_test[:1000], y_test[:1000])
    print(f"Trained accuracy: {accuracy_trained}")
    print(nn.weights_1_2)

if __name__ == '__main__':
    benchmark()