import sys

from simple_net import SimpleNet, load_mnist, save_checkpoint, load_checkpoint


def benchmark(model_id=None):
    ((x_train, y_train), (x_test, y_test), _) = load_mnist()

    if not model_id:
        nn = SimpleNet(alpha=0.0001)
        error, accuracy = nn.test(x_test[:1000], y_test[:1000])
        nn.train(x_train, y_train)
        model_id = save_checkpoint(nn)
    else:
        nn = load_checkpoint(model_id)

    error_trained, accuracy_trained = nn.test(x_test, y_test)

    print(f"model_id: {model_id}")
    print(f"Trained error: {error_trained}, accuracy: {accuracy_trained}")


if __name__ == "__main__":

    if len(sys.argv) == 2:
        benchmark(sys.argv[1])
    else:
        benchmark()
