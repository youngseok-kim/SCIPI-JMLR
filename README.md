# SCIPI-JMLR

## Introduction

This is for introduction and reproduction of the paper

Cheolmin Kim, Youngseok Kim, Diego Klabjan, Scale Invariant Power Iteration (2019, arXiv)

which is submitted to Journal of Machine Learning Research (JMLR).

We do not add arXiv link but you can quickly search it in google.

We will add a JMLR link to the paper when it is accepted and published.

## Repository

### Tutorial

The main application of the methods are Kullback-Leibler diverence Non-negative Matrix Factorization (KLNMF)

and its subproblem (KLNMF subproblem)

We provide python tutorials in `notebooks` as this is one of the most frequently used languages in Machine Learning

The original simulations and experiments are based on Julia programming language.

Even though random seeds are different, we can easily reproduce similar results

(e.g. qualitatively having the same take-away messages, or similar conclusion in relative comparison)

with other languagues such as Python3.


#### WT like real data example (python3)

https://github.com/youngseok-kim/SCIPI-JMLR/tree/main/notebooks/klnmf_real_world_dense_example.ipynb

#### KOS like real data example (python3)

https://github.com/youngseok-kim/SCIPI-JMLR/tree/main/notebooks/klnmf_real_world_sparse_example.ipynb

#### Synthetic data examples (python3)

https://github.com/youngseok-kim/SCIPI-JMLR/tree/main/notebooks/klnmf_large_synthetic.ipynb

https://github.com/youngseok-kim/SCIPI-JMLR/tree/main/notebooks/klnmf_large_synthetic.ipynb

#### Julia codes

https://github.com/youngseok-kim/SCIPI-JMLR/tree/main/julia

### Data

For KL-NMF, we downloaded the data from

https://www.microsoft.com/en-us/research/project,
https://archive.ics.uci.edu/ml/datasets/bag+of+words,
and
https://snap.stanford.edu/data/wiki-Vote.html

For KL-NMF subproblem, we downloaded the data from

https://github.com/stephenslab/mixsqp-paper/tree/master/data

This data set may not be available in recent commits.

Please see `data`

For GMM, we use R mlbench package for the data sets.

For ICA, we downloaded the data from

https://archive.ics.uci.edu/ml/index.php

### Julia codes

Julia codes used in the paper can be found in `julia`.

## Copyright

### Author

Cheolmin Kim, Youngseok Kim, Diego Klabjan