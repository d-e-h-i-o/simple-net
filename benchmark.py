from simple_net import SimpleNet, load_mnist

def benchmark():
    ((x_train, y_train), (x_test, y_test), _) = load_mnist()

    nn = SimpleNet(alpha=0.0001)
    error, accuracy = nn.test(x_test[:1000], y_test[:1000])
    print(f"Untrained error: {error}, accuracy: {accuracy}")

    for _ in range(20):
        nn.train(x_train, y_train)

    error_trained, accuracy_trained = nn.test(x_test[:1000], y_test[:1000])
    print(f"Trained error: {error_trained}, accuracy: {accuracy_trained}")

if __name__ == '__main__':
    benchmark()