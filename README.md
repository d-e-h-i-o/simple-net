# SimpleNet

A very simple, three layered neural network, as excersice while reading `Grokking Deep Learning` by Andrew W. Trask. Goal is to evaluate it on MNIST, and to add features as I progress in the book.

# Performance

| model_id                             | alpha | iterations | hidden_layer_size | dropout | error | accuracy |
|--------------------------------------|-------|------------|-------------------|---------|-------|----------|
| 0cd32548-f205-11eb-a7dc-7c04d0d54c7a | 0.005 | 20         | 264               | no      | 0.18  | 0.945    |

# Usage

Run `python benchmark.py [model_id]`
