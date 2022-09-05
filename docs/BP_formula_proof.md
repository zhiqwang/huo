---
layout: post
title: The proof of backpropagation algorithm
tags: [deep-learning, backpropagation]
date: 2018-04-30 10:00:00 +0800
---

We have the feed-forward network form as
$$a^{l} = \sigma\left(w^l a^{l - 1} + b^l\right),$$
and the weighted input $z$ as
$$z^l = w^l a^{l - 1} + b^l.$$

Define the error $\delta^l_j$ in the $j^{\rm th}$ entries of layer $l$ as
$$\delta^l_j := \frac{\partial C}{\partial z^l_j}.$$

### Backpropagation algorithm
The way compute the gradient of $C$ with variable $w^l_{jk}$ and $b^l_j$.

1. The error of the output layer: $$\delta^L = \nabla_a C \odot \sigma^{\prime}\left(z^L\right). \tag{BP1}$$
2. Compute the error of the current layer in terms of the error in the next layer: $$\delta^l = \left(\left(w^{l+1}\right)^{\mathsf T} \delta^{l+1}\right) \odot \sigma^{\prime}\left(z^l\right). \tag{BP2}$$
3. Rate of change of the cost with respect to $b_j$ in the layer $l$: $$\frac{\partial C}{\partial b^l_j} = \delta^l_j. \tag{BP3}$$
4. Rate of change of the cost with respect to $w_{jk}$ in the layer $l$: $$\frac{\partial C}{\partial w^l_{jk}} = a^{l - 1}_k \delta^l_j. \tag{BP4}$$

_Proof_.

- The error of the output layer: $$\delta^L_j = \frac{\partial C}{\partial z^L_j} = \sum_k \frac{\partial C}{\partial a^L_k} \frac{\partial a^L_k}{\partial z^L_j} = \frac{\partial C}{\partial a^L_j}\,\sigma^{\prime}\left(z^L_j\right).$$
- Compute the error of the current layer with respect to next layer: $$\begin{align} \delta^l_j &= \frac{\partial C}{\partial z^l_j} \\ &= \sum_k \frac{\partial C}{\partial z^{l+1}_k} \frac{\partial z^{l+1}_k}{\partial z^l_j} \\ &= \sum_k \delta^{l+1}_k \frac{\partial z^{l+1}_k}{\partial z^l_j} \\ &= \sum_k \delta^{l+1}_k \left(\sum_m \frac{\partial z^{l+1}_k}{\partial a^l_m} \frac{\partial a^l_m}{\partial z^l_j} \right) \\ &= \sum_k \delta^{l+1}_k \left(\frac{\partial z^{l+1}_k}{\partial a^l_j}\,\sigma^{\prime}\left(z^l_j\right)\right) \\ &= \sum_k \delta^{l+1}_k \left(\frac{\partial \left(\sum_n w^{l+1}_{kn} a^l_n + b^{l+1}_k\right)}{\partial a^l_j}\,\sigma^{\prime}\left(z^l_j\right)\right) \\ &= \left(\sum_k w^{l+1}_{kj} \delta^{l+1}_k\right)\,\sigma^{\prime}\left(z^l_j\right). \end{align}$$
- Gradient with respect to $b^l_j$: $$\frac{\partial C}{\partial b^l_j} = \sum_k \frac{\partial C}{\partial z^l_k} \frac{\partial z^l_k}{\partial b^l_j} = \sum_k \frac{\partial \left(\sum_n w^l_{kn}a^{l - 1}_n + b^l_k\right)}{\partial b^l_j} \delta^l_k = \delta^l_j.$$
- Gradient with respect to $w^l_{jk}$: $$ \frac{\partial C}{\partial w^l_{jk}} = \sum_m \frac{\partial C}{\partial z^l_m} \frac{\partial z^l_m}{\partial w^l_{jk}} = \sum_m \frac{\partial \left(\sum_n w^l_{mn}a^{l - 1}_n + b^l_m \right)}{\partial w^l_{jk}} \delta^l_m = a^{l-1}_k \delta^l_j$$
