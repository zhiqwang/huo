---
layout: post
title: Backpropagation In Convolutional Neural Networks
tags: [deep-learning, backpropagation, convolution, pooling]
date: 2018-05-25 18:00:00 +0800
---

## Convolutional Neural Networks

Neuons in CNNs share weights unlike in MLPs where each neuron has a separate weight vector. This sharing of weights ends up reducing the overall number of trainable weights hence introducing sparsity.

### Convolution
Given an input image $I$ of dimension $H \times W$ and a filter (kernel) $K$ of dimension $k_1 \times k_2$, the convolution operation is given by:

$$\begin{align}
\left(I \ast K \right)_{ij} &= \sum^{k_1 - 1}_{m = 0} \sum^{k_2 - 1}_{n = 0} I(i - m, j - n)K(m,n) \tag{2}\\
&= \sum^{k_1 - 1}_{m = 0} \sum^{k_2 - 1}_{n = 0} I(i + m, j + n) K(-m, -n) \tag{3}
\end{align}$$

### Forward Propagation

$$z^l = {\rm rot}_{180^\circ} \left\{ w^l \right\} \ast a^{l-1} + b^l \tag {6}$$

$$z^l_{i,j} = \sum^{k_1-1}_{m=0} \sum^{k_2-1}_{n=0} w^l_{m,n} a^{l-1}_{i+m,j+n} + b^l_{i,j} \tag{7}$$

$$a^l_{i,j} = \sigma\left(z^l_{i,j}\right)$$

### Backward Propagation

We all know the forward pass of a convolutional layer uses convolutions. But, the backward pass during backpropagation also uses convolutions.

Each weight in the filter contributes to each pixel in the output map. Thus, any change in a weight in the filter will affect all the output pixels. Thus, all these changes add up to contribute to the final loss.

$$\begin{align}
\frac{\partial C}{\partial w^l_{m,n}} &= \sum^{H-k_1}_{i=0}\sum^{W-k_2}_{j=0} \frac{\partial C}{\partial z^l_{i,j}}\frac{\partial z^l_{i,j}}{\partial w^l_{m,n}} \\
&= \sum^{H-k_1}_{i=0}\sum^{W-k_2}_{j=0} \delta^l_{i,j} \, \frac{\partial z^l_{i,j}}{\partial w^l_{m,n}} \\
&= \sum^{H-k_1}_{i=0}\sum^{W-k_2}_{j=0} \delta^l_{i,j} \, \frac{\partial}{\partial w^l_{m,n}} \left(\sum^{k_1-1}_{m^{\prime}=0} \sum^{k_2-1}_{n^{\prime}=0} w^l_{m^{\prime},n^{\prime}} a^{l-1}_{i+m^{\prime},j+n^{\prime}} + b^l_{i^{\prime},j^{\prime}}\right) \\
&= \sum^{H-k_1}_{i=0}\sum^{W-k_2}_{j=0} \delta^l_{i,j} \, a^{l-1}_{i+m,j+n}, \tag{13}
\end{align}$$

In other words,

$$\frac{\partial C}{\partial w^l} = {\rm rot}_{180^{\circ}}\left\{ \delta^l \right\} \ast a^{l-1} \tag{14}$$

The dual summation in Eqn.13 is as a result of weight sharing in the network (same weight kernel is slid over all of the input feature map during convolution).

The region in the output affected by pixel $z_{i^{\prime},j^{\prime}}$ from the input is the region in the output bounded by the box $Q$ where the top left corner pixel is given by $\left(i^{\prime} - k_1 + 1, j^{\prime} - k_2 + 1\right)$ and the bottom right corner pixel is given by $\left(i^{\prime},j^{\prime}\right)$.

$$\begin{align}
\frac{\partial C}{\partial z^l_{i^{\prime},j^{\prime}}} &= \sum_{i,j \in Q} \frac{\partial C}{\partial z^{l+1}_{i,j}}\frac{\partial z^{l+1}_{i,j}}{\partial z^l_{i^{\prime},j^{\prime}}} \\
&= \sum_{i,j \in Q} \frac{\partial C}{\partial z^{l+1}_{i,j}}\frac{\partial z^{l+1}_{i,j}}{\partial a^l_{i^{\prime},j^{\prime}}} \sigma^{\prime}\left(z^l_{i^{\prime},j^{\prime}}\right) \\
&= \sum_{i,j \in Q} \delta^{l+1}_{i,j} \, \frac{\partial}{\partial a^l_{i^{\prime},j^{\prime}}} \left(\sum^{k_1-1}_{m=0} \sum^{k_2-1}_{n=0} w^{l+1}_{m,n} a^l_{i+m,j+n} + b^{l+1}_{i,j} \right) \sigma^{\prime}\left(z^l_{i^{\prime},j^{\prime}}\right) \\
&= \sum^{i^{\prime}}_{i = i^{\prime} - k_1 + 1} \sum^{j^{\prime}}_{j = j^{\prime} - k_2 + 1} \delta^{l+1}_{i,j} \, \frac{\partial}{\partial a^l_{i^{\prime},j^{\prime}}} \left(\sum^{k_1-1}_{m=0} \sum^{k_2-1}_{n=0} w^{l+1}_{m,n} a^l_{i+m,j+n} + b^{l+1}_{i,j} \right) \sigma^{\prime}\left(z^l_{i^{\prime},j^{\prime}}\right) \\
&= \sum^{i^{\prime}}_{i = i^{\prime} - k_1 + 1} \sum^{j^{\prime}}_{j = j^{\prime} - k_2 + 1} \delta^{l+1}_{i,j} \, w^{l+1}_{i^{\prime} - i, j^{\prime} - j} \, \sigma^{\prime}\left(z^l_{i^{\prime},j^{\prime}}\right) \\
&= \sum^{k_1 - 1}_{m = 0} \sum^{k_2 - 1}_{n = 0} \delta^{l+1}_{i^{\prime} - m, j^{\prime} - n} \, w^{l+1}_{m,n} \,  \sigma^{\prime}\left(z^l_{i^{\prime},j^{\prime}}\right)
\end{align}$$

In other words,

$$\frac{\partial C}{\partial z^l} = \left\{\delta^{l+1} \ast w^{l+1} \right\} \odot \sigma^{\prime}\left(z^l\right)$$

### [Pooling Layer](http://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/)
The function of the pooling layer is to progressively reduce the spatial size of the representation to reduce the amount of parameters and computation in the network, and hence to also control overfitting.

Pooling units are obtained using functions like max-pooling, average pooling and even $L_2$-norm pooling. At the pooling layer, forward propagation results in an $N \times N$ pooling block being reduced to a single value - value of the "winning unit". Backpropagation of the pooling layer then computes the error which is acquired by this single value "winning unit". Backpropagation of the pooling layer then computes the error which is acquired by this single value "winning value".

To keep track of the "winning unit" its index noted during the forward pass and use for gradient routing during backpropagation. Gradient routing is done in the following ways:

- **Max-pooling** - the error is just assigned to where it comes from - the "winning unit" because other units in the previous layer's pooling blocks did not contribute to it hence all the other assigned values of zero.
- **Average pooling** - the error is multiplied by $1/ (N\times N)$ and assigned to the whole pooling block (all units get this same value).
