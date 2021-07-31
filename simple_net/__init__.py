__version__ = "0.1.0"

from .simple_net import SimpleNet

import os
import requests
import pickle
import gzip
import uuid


def download_mnist():

    path = "./minst.pkl.gz"
    url = "https://github.com/pytorch/tutorials/raw/master/_static/mnist.pkl.gz"

    if not os.path.exists(path):
        content = requests.get(url).content
        with open(path, "wb") as file:
            file.write(content)


def load_mnist():
    """Use like this:
    ((x_train, y_train), (x_test, y_test), _) = load_minst()
    """
    download_mnist()

    with gzip.open("./minst.pkl.gz", "rb") as file:
        return pickle.load(file, encoding="latin-1")


def save_checkpoint(model):
    model_id = str(uuid.uuid1())
    with open(f"models/{model_id}", "wb") as file:
        file.write(pickle.dumps(model))

    return model_id


def load_checkpoint(model_id):
    with open(f"models/{model_id}", "rb") as file:
        return pickle.load(file)
