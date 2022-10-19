# On Classification Thresholds for Graph Attention with Edge Features

The directory `src` contains the source code for our paper: [On Classification Thresholds for Graph Attention with Edge Features
](https://arxiv.org/abs/2210.10014).

## Abstract
The recent years we have seen the rise of graph neural networks for prediction tasks on graphs. One of the dominant architectures is graph attention due to its ability to make predictions using weighted edge features and not only node features. In this paper we analyze, theoretically and empirically, graph attention networks and their ability of correctly labelling nodes in a classic classification task. More specifically, we study the performance of graph attention on the classic contextual stochastic block model (CSBM). In CSBM the nodes and edge features are obtained from a mixture of Gaussians and the edges from a stochastic block model. We consider a general graph attention mechanism that takes random edge features as input to determine the attention coefficients. We study two cases, in the first one, when the edge features are noisy, we prove that the majority of the attention coefficients are up to a constant uniform. This allows us to prove that graph attention with edge features is not better than simple graph convolution for achieving perfect node classification. Second, we prove that when the edge features are clean graph attention can distinguish intra- from inter-edges and this makes graph attention better than classic graph convolution.

## Disclaimer
Please note that the code is rather crude at the moment and not documented very well, however, it works and produces the results as claimed in the paper.
