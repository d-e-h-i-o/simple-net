# SimpleNet

A very simple, three layered neural network, as excercise while reading `Grokking Deep Learning` by Andrew W. Trask. Goal is to evaluate it on MNIST, and to add features as I progress in the book.

# Benchmark

| model_id                             | alpha | iterations | hidden_layer_size | dropout | error | accuracy |
|--------------------------------------|-------|------------|-------------------|---------|-------|----------|
| 0cd32548-f205-11eb-a7dc-7c04d0d54c7a | 0.005 | 20         | 264               | no      | 0.18  | 0.945    |
| 524cdfd4-f227-11eb-ae71-7c04d0d54c7a | 0.05  | 350        | 264               | no      | 0.09  | 0.974    |
| 14d06eb2-f300-11eb-86a7-7c04d0d54c7a | 0.005  | 20        | 264               | yes      | 0.095 | 0.972    |

I'm a bit suspicious of the reported accuracy, since it is considerable higher than in the book (where 0.81 test accuracy is reported, with a model with hidden_size=100, dropout, alpha 0.005 and 300 iterations). I'll have to make some more runs to find out why.

# Usage

Run `python benchmark.py [model_id]` to re-run a benchmark of a certain model, or `python benchmark.py` for a run with the current configurations in `benchmark.py`.
