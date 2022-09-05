---
layout: post
title: Improving the way neural networks learn - Problem
tags: [deep-learning, backpropagation]
date: 2018-05-02 21:00:00 +0800
---

### Softmax

$$a^L_j = \frac{e^{z^L_j}}{\sum^n_i e^{z^L_i}},\tag{78}$$

#### Monotonicity of softmax

When $j = k$,
$$\begin{align}\frac{\partial a^L_j}{\partial z^L_k}
&= \frac{\left(\frac{\partial e^{z^L_k}}{\partial z^L_k}\right) \cdot \left(\sum^n_i e^{z^L_i}\right) - \left(\frac{\partial \sum^n_i e^{z^L_i}}{\partial z^L_k}\right)\cdot\left(e^{z^L_k}\right)}{\left(\sum^n_i e^{z^L_i}\right)^2} \\
&= \frac{\left(e^{z^L_k}\right) \cdot \left(\sum^n_i e^{z^L_i}\right) - \left(e^{z^L_k}\right)\cdot\left(e^{z^L_k}\right)}{\left(\sum^n_i e^{z^L_i}\right)^2} \\
&= \frac{e^{z^L_k} \cdot \left(\sum^n_i e^{z^L_i} - e^{z^L_k} \right)}{\left(\sum^n_i e^{z^L_i}\right)^2} > 0\end{align}$$

When $j \neq k$,
$$\begin{align}\frac{\partial a^L_j}{\partial z^L_k}
&= \frac{- \left(\frac{\partial \sum^n_i e^{z^L_i}}{\partial z^L_k}\right)\cdot e^{z^L_j}}{\left(\sum^n_i e^{z^L_i}\right)^2} \\
&= \frac{- e^{z^L_k}\cdot e^{z^L_j}}{\left(\sum^n_i e^{z^L_i}\right)^2} < 0
\end{align}$$

As a consequence, increasing $z^L_j$ is guaranteed to increase the corresponding output activation, $a^L_j$, and will decrease all the other output activations.

#### Backpropagation Algorithm

$$C \equiv - \ln a^L_y. \tag{80}$$

$$\begin{align}
\frac{\partial C}{\partial a^L_y} &= - \frac{1}{a^L_y} \\
&= - \frac{\sum^n_i e^{z^L_i}}{e^{z^L_y}}
\end{align}$$

if $j = y$, i.e., $y_j = 1$, then
$$\begin{align}
\delta^L_j &= \sum^n_i \frac{\partial C}{\partial a^L_i} \cdot \frac{\partial a^L_i}{\partial z^L_j} \\
&= \frac{\partial C}{\partial a^L_y} \cdot \frac{\partial a^L_y}{\partial z^L_j} \\
&= - \frac{\sum^n_i e^{z^L_i}}{e^{z^L_j}} \cdot \frac{e^{z^L_j} \cdot \left(\sum^n_i e^{z^L_i} - e^{z^L_j} \right)}{\left(\sum^n_i e^{z^L_i}\right)^2} \\
&= - \frac{\sum^n_i e^{z^L_i} - e^{z^L_j}}{\sum^n_i e^{z^L_i}} \\
&= \frac{e^{z^L_j}}{\sum^n_i e^{z^L_i}} - 1 \\
&= a^L_j - y_j
\end{align}$$

if $j \neq y$, i.e., $y_j = 0$, then
$$\begin{align}
\delta^L_j &= \sum^n_i \frac{\partial C}{\partial a^L_i} \cdot \frac{\partial a^L_i}{\partial z^L_j} \\
&= \frac{\partial C}{\partial a^L_y} \cdot \frac{\partial a^L_y}{\partial z^L_j} \\
&= - \frac{\sum^n_i e^{z^L_i}}{e^{z^L_y}} \cdot \frac{- e^{z^L_j}\cdot e^{z^L_y}}{\left(\sum^n_i e^{z^L_i}\right)^2} \\
&= \frac{e^{z^L_j}}{\sum^n_i e^{z^L_i}} \\
&= \frac{e^{z^L_j}}{\sum^n_i e^{z^L_i}} - 0 \\
&= a^L_j - y_j
\end{align}$$

**Where does the "softmax" name come from?** Suppose we change the softmax function so the output activations are given by
$$a^L_j = \frac{e^{c z^L_j}}{\sum_k e^{c z^L_k}},\tag{83}$$
where $c$ is a positive constant.

Suppose we allow $c$ to become large, i.e., $c\to\infty$. What is the limiting value for the output activations $a^L_j$?

_Proof_. Suppose that $z_1 \leq z_2 \leq \cdots \leq z_n$, then,
$$\begin{align}
\lim_{c\to\infty} a^L_j &= \lim_{c\to\infty} \frac{e^{c z^L_j}}{\sum_k e^{c z^L_k}} \\
&= \lim_{c\to\infty} \frac{1}{e^{c\left(z^L_1-z^L_j\right)} + e^{c\left(z^L_2-z^L_j\right)} + \cdots + e^{c\left(z^L_j -z^L_j\right)} + \cdots + e^{c\left(z^L_n-z^L_j\right)}},
\end{align}$$

Then $\lim_{c\to\infty} a^L_j = 0$ for $j < n$, and $\lim_{c\to\infty} a^L_n = 1$.

In conclusion,
$$\lim_{c\to\infty} \begin{bmatrix} a_1 \\ a_2 \\ \vdots \\ a_n \end{bmatrix} = \begin{bmatrix} 0 \\ \vdots \\ 0 \\ 1 \end{bmatrix}$$

