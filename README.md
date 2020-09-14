# ActivationRelaxation

# Activation Relaxation
Code for the paper "Activation Relaxation: A Local Dynamical Approximation to Backprop in the Brain" -- https://arxiv.org/pdf/2009.05359.pdf.

The paper demonstrates a novel algorithm which can converge to the backpropagation of error algorithm using only local learning rules. We demonstrate numerical convergence to autodiff gradients as well as strong performance in training MLPs on MNIST and Fashion-MNIST. Moreover, we propose additional simplifications of the algorithm which remove further biological implausibilities and which maintain strong learning performance.

## Installation and Usage
Simply `git clone` the repository to your home computer. The `numerical_test.py` file will recreate the numerical results in Figure 1. The `main.py` file contains code to reproduce all the other experiments.

## Requirements 

The code is written in [Python 3.x] and uses the following packages:
* [NumPY]
* [PyTorch] version 1.3.1
* [matplotlib] for plotting figures

If you find this code useful, please reference in your paper:
```
@article{millidge2020activation,
  title={Activation Relaxation: A Local Dynamical Approximation to Backprop in the Brain},
  author={Millidge, Beren and Tschantz, Alexander and Seth, Anil and Buckley, Christopher},
  journal={arXiv preprint arXiv:2009.05359},
  year={2020}
}
```
