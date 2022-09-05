---
layout: post
title: How the backpropagation algorithm works
tags: [deep-learning, backpropagation]
date: 2018-04-26 00:00:00 +0800
---

## Using neural nets to recognize handwritten digits
- Indeed, it's the smoothness of the $\sigma$ function that is the crucial fact, not its detailed form. The smoothness of $\sigma$ means that small changes $\Delta w_j$ in the weights and $\Delta b$ in the bias will produce a small change $\Delta \mbox{output}$ in the output from the neuron. In fact, calculus tells us that $\Delta \mbox{output}$ is well approximated by $$\Delta \mbox{output} \approx \sum_j \frac{\partial \, \mbox{output}}{\partial w_j} \Delta w_j + \frac{\partial \, \mbox{output}}{\partial b} \Delta b. \tag{5}$$

## How the backpropagation algorithm works
- Backpropagation isn't just a fast algorithm for learning. It actually gives us detailed insights into how changing the weights and biases changes the overall behaviour of the network.
- The demon sits at the $j^\rm{th}$ neuron in layer $l$. As the input to the neuron comes in, the demon messes with the neuron's operation. It adds a little change $\Delta z^l_j$ to the neuron's weighted input, so that instead of outputting $\sigma\left(z^l_j\right)$, the neuron instead outputs $\sigma\left(z^l_j+\Delta z^l_j\right)$. This change propagates through later layers in the network, finally causing the overall cost to change by an amount $\frac{\partial C}{\partial z^l_j} \Delta z^l_j$.
- The four fundamental equations behind backpropagation, we define the error $\delta^l_j$ of neuron $j$ in layer $l$ by $$\delta^l_j \equiv \frac{\partial C}{\partial z^l_j}. \tag{29}$$

    1. An equation for the error in the output layer, $\sigma^L$: The components of $\delta^L$ are given by $$\delta^L = \nabla_a C \odot \sigma^{\prime}\left(z^L\right) \tag{BP1}$$
    2. An equation for the error $\delta^l$ in terms of the error in the next layer, $\delta^{l+1}$: In particular $$\delta^l = \left(\left(w^{l+1}\right)^{\mathsf T} \delta^{l+1}\right) \odot \sigma^{\prime}\left(z^l\right). \tag{BP2}$$
    3. An equation for the rate of change of the cost with respect to any bias in the network: In particular: $$\frac{\partial C}{\partial b^l} = \delta^l. \tag{BP3}$$
    4. An equation for the rate of change of the cost with respect to any weight in the network: In particular: $$\frac{\partial C}{\partial w^l} = \delta^l \cdot\left(a^{l-1}\right)^{\mathsf T}. \tag{BP4}$$

- Summing up, we've learnt that a weight will learn slowly if either the input neuron is low-activation, or if the output neuron has saturated, i.e., is either high- or low-activation.
- You can think of the backpropagation algorithm as providing a way of computing the sum over the rate factor for all these paths. Or, to put it slightly differently, the backpropagation algorithm is a clever way of keeping track of small perturbations to the weights (and bias) as they propagate through the network, reach the output, and then affect the cost.

### The Backpropagation algorithm over a mini-batch
We can begin with a matrix $X = [x_1, x_2, \ldots, x_m]$ whose columns are the vectors in the mini-batch. We forward-propagate by multiplying by the weight matrices, adding a suitable matrix for the bias terms, and applying the sigmoid function everywhere. We backpropagate along similar lines.

- **Input a set of training examples**
- **For mini-batch examples $X = [x_1, x_2, \ldots, x_m]$**: Set the corresponding input activation $A^1 = X$, and perform the following steps:

    + **Feedforward**: For each $l = 2, 3,\ldots, L$ compute $$Z^l = w^l A^{l-1} + b^l,$$ and $$A^l = \sigma\left(Z^l\right).$$
    + **Output error $\delta^L$**: Compute the vector $$\delta^L = \nabla_a C \odot \sigma^{\prime}\left(Z^L\right).$$ 
    + **Backpropagate the error**: For each $l = L - 1, L - 2, \ldots, 2$ compute $$\delta^l = \left(\left(w^{l+1}\right)^{\mathsf T} \delta^{l+1}\right)\odot\sigma^{\prime}\left(Z^l\right).$$

- **Gradient descent**: For each $l = L, L - 1, \ldots, 2$ update the weights according to the rule $$w^l \to w^l - \frac{\eta}{m} \delta^l\left(A^{l-1}\right)^{\mathsf T},$$ and the biases according to the rule $$b^l \to b^l - \frac{\eta}{m} \sum_{\rm row} \delta^l.$$
