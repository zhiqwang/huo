---
layout: post
title: Backpropagation algorithm and realization
tags: [deep-learning, backpropagation]
date: 2018-05-01 17:00:00 +0800
---

We have the feed-forward network form as
$$a^l = \sigma\left(w^l a^{l - 1} + b^l\right),$$
and the weighted input $z$ as
$$z^l = w^l a^{l - 1} + b^l.$$

Define the error $\delta^l_j$ in the $j^{\rm th}$ entries of layer $l$ as
$$\delta^l_j := \frac{\partial C}{\partial z^l_j}.$$

### The four fundamental equations behind backpropagation
1. The error of the output layer: $$\delta^L = \nabla_a C \odot \sigma^{\prime}\left(z^L\right). \tag{BP1}$$
2. Compute the error of the current layer in terms of the error in the next layer: $$\delta^l = \left(\left(w^{l+1}\right)^{\mathsf T} \delta^{l+1}\right) \odot \sigma^{\prime}\left(z^l\right). \tag{BP2}$$
3. Rate of change of the cost with respect to $b^l$: $$\frac{\partial C}{\partial b^l} = \delta^l. \tag{BP3}$$
4. Rate of change of the cost with respect to $w^l$: $$\frac{\partial C}{\partial w^l} = \delta^l \cdot\left(a^{l-1}\right)^{\mathsf T}. \tag{BP4}$$

### The Backpropagation algorithm

- **Input a set of training examples**
- **For each training example $x$**: Set the corresponding input activation $a^{x,1}$, and perform the following steps:

    + **Feedforward**: For each $l = 2, 3,\ldots, L$ compute $$z^{x,l} = w^l a^{x,l-1} + b^l,$$ and $$a^{x,l} = \sigma\left(z^{x,l}\right).$$
    + **Output error $\delta^{x,L}$**: Compute the vector $$\delta^{x,L} = \nabla_a C_x \odot \sigma^{\prime}\left(z^{x,L}\right).$$ 
    + **Backpropagate the error**: For each $l = L - 1, L - 2, \ldots, 2$ compute $$\delta^{x,l} = \left(\left(w^{l+1}\right)^{\mathsf T} \delta^{x,l+1}\right)\odot\sigma^{\prime}\left(z^{x,l}\right).$$

- **Gradient descent**: For each $l = L, L - 1, \ldots, 2$ update the weights according to the rule $$w^l \to w^l - \frac{\eta}{m}\sum_x \delta^{x,l}\left(a^{x,l-1}\right)^{\mathsf T},$$ and the biases according to the rule $$b^l \to b^l - \frac{\eta}{m} \sum_x \delta^{x,l}.$$

### The Backpropagation algorithm over a mini-batch
We can begin with a matrix $X = [x_1, x_2, \ldots, x_m]$ whose columns are the vectors in the mini-batch. We forward-propagate by multiplying by the weight matrices, adding a suitable matrix for the bias terms, and applying the sigmoid function everywhere. We backpropagate along similar lines.

- **Input a set of training examples**
- **For mini-batch examples $X = [x_1, x_2, \ldots, x_m]$**: Set the corresponding input activation $A^1 = X$, and perform the following steps:

    + **Feedforward**: For each $l = 2, 3,\ldots, L$ compute $$Z^l = w^l A^{l-1} + b^l,$$ and $$A^l = \sigma\left(Z^l\right).$$
    + **Output error $\delta^L$**: Compute the vector $$\delta^L = \nabla_a C \odot \sigma^{\prime}\left(Z^L\right).$$ 
    + **Backpropagate the error**: For each $l = L - 1, L - 2, \ldots, 2$ compute $$\delta^l = \left(\left(w^{l+1}\right)^{\mathsf T} \delta^{l+1}\right)\odot\sigma^{\prime}\left(Z^l\right).$$

- **Gradient descent**: For each $l = L, L - 1, \ldots, 2$ update the weights according to the rule $$w^l \to w^l - \frac{\eta}{m} \delta^l\left(A^{l-1}\right)^{\mathsf T},$$ and the biases according to the rule $$b^l \to b^l - \frac{\eta}{m} \sum_{\rm row} \delta^l.$$

An example:

- Suppose batch size is $50$, $L = 3$, neural network as $[784, 30, 10]$,
- $w^2$ with dimension $30 \times 784$, $w^3$ with dimension $10 \times 30$,
- $b^2$ with dimension $30 \times 1$, $b^3$ with dimension $10 \times 1$,
- $X, A^1$ with dimension $784\times 50$,
- $Z^2, A^2, \delta^2$ with dimension $30 \times 50$, $Z^3, A^3, \delta^3$ with dimension $10 \times 50$,

    + Backpropagate the error: $l = 2$, we have $$\delta^2 = \left(\left(w^3\right)^{\mathsf T} \delta^3\right)\odot\sigma^{\prime}\left(Z^2\right).$$
    + Gradient descent: $l = 3$, $$w^3 \to w^3 - \frac{\eta}{m} \delta^3\left(A^2\right)^{\mathsf T},$$ and $$b^3 \to b^3 - \frac{\eta}{m} \sum_{\rm row} \delta^3.$$
