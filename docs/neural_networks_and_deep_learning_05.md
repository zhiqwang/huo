---
layout: post
title: Why are deep neural networks hard to train?
tags: [deep-learning, universality]
date: 2018-05-17 11:00:00 +0800
---

## Why are deep neural networks hard to train?

### The vanishing gradient problem
- Recall that we randomly initialized the weight and bias in the network. It is extremely unlikely our initial weights and bias will do a good job at whatever it is we want out network to do. The random initialization means the first layer throws away most information about the input image. Even if later layers have been extensively trained, they will still find it extremely difficult to identify the input image, simply because they don't have enough information. And so it can't possibly be the case that not much learning needs to be done in the first layer.

### What's causing the vanishing gradient problem? Unstable gradients in deep neural nets
- To understand why the vanishing gradient problem occurs, let's explicitly write out the entire expression for the gradient: $$\frac{\partial C}{\partial b_1} = \sigma^{\prime}(z_1) \, w_2 \sigma^{\prime}(z_2) \, w_3 \sigma^{\prime}(z_3) \, w_4 \sigma^{\prime}(z_4) \, \frac{\partial C}{\partial a_4}. \tag{122}$$
- Excepting the very last term, this expression is a product of terms of the form $w_j \sigma^{\prime}\left(z_j\right)$. The derivative of the function $\sigma$ reaches a maximum at $\sigma^{\prime}(0) = 1/4$. Now, if we use out standard approach to initializing the weights in the network, then we'll choose the weights using a Gaussian with mean $0$ and standard deviation $1$. So the weights will usually satisfy $\left\vert w_j \right\vert < 1$. Putting these observations together, we see that the terms $w_j\sigma^{\prime}\left(z_j\right)$ will usually satisfy $\left\vert w_j\sigma^{\prime}\left(z_j\right)\right\vert < 1/4$. And when we take a product of many such terms, the product will tend to exponentially decrease: the more terms, the smaller the product will be. This is starting to smell like a possible explanation for the vanishing gradient problem.
- The fundamental problem here isn't so much the vanishing gradient problem or the exploding gradient problem. It's that the gradient in early layers is the product of terms from all the later layers. When there are many layers, that's an intrinsically unstable situation. The only way all layers can learn at close to the same speed is if all those products of terms come close to balancing out. Without some mechanism or underlying reason for that balancing to occur, it's highly unlikely to happen simply by chance. In short, the real problem here is that neural networks suffer from an _unstable gradient problem_. As a result, if we use standard gradient-based learning techniques, different layers in the network will tend to learn at wildly different speed.
- The prevalence of the vanishing gradient problem. We've seen that the gradient can either vanish or explode in the early layers of a deep network. In fact, when using sigmoid neurons the gradient will usually vanish.

Consider the product

$$\left\vert w\sigma^{\prime}\left(wa+b\right)\right\vert.$$

Suppose

$$\left\vert w\sigma^{\prime}\left(wa+b\right)\right\vert \geq 1.$$

- Argue that this can only ever occur if $\left\vert w \right\vert \geq 4$.

$$\begin{align}
\sigma^{\prime}(z) &= \sigma(z) \cdot (1 - \sigma(z)) \leq \frac{1}{4}
\end{align}$$

- Supposing that $\left\vert w \right\vert \geq 4$, consider the set of input activations $a$ for which $\left\vert w \sigma^{\prime}(wa + b)\right\vert \geq 1$. Show that the set of $a$ satisfying that constraint can range over an interval no greater in width than

$$\frac{2}{\left\vert w\right\vert} \ln\left(\frac{\left\vert w\right\vert \left(1+\sqrt{1-4/\left\vert w\right\vert}\right)}{2}-1\right). \tag{123}$$

$$\sigma(z)\cdot\left(1 - \sigma(z)\right) \geq \frac{1}{\left\vert w \right\vert}$$

$$\sigma^2 - \sigma + 1 / \left\vert w \right\vert = 0$$

$$\frac{1 - \sqrt{1 - 4/\left\vert w \right\vert}}{2} \leq \sigma \leq \frac{1 + \sqrt{1 - 4/\left\vert w \right\vert}}{2}$$

$$\frac{1 - \sqrt{1 - 4/\left\vert w \right\vert}}{2} \leq \frac{1}{1 + e^{-z}} \leq \frac{1 + \sqrt{1 - 4/\left\vert w \right\vert}}{2}$$

$$\begin{align}
1 + e^{-z} &\leq \frac{2}{1 - \sqrt{1 - 4/\left\vert w \right\vert}} \\
e^{-z} &\leq \frac{2}{1 - \sqrt{1 - 4/\left\vert w \right\vert}} - 1 \\
&\leq \frac{\left\vert w \right\vert \left(1 + \sqrt{1 - 4/\left\vert w \right\vert}\right)}{2} - 1 \\
- z &\leq \ln\left(\frac{\left\vert w \right\vert \left(1 + \sqrt{1 - 4/\left\vert w \right\vert}\right)}{2} - 1 \right) \\
z &\geq - \ln\left(\frac{\left\vert w \right\vert \left(1 + \sqrt{1 - 4/\left\vert w \right\vert}\right)}{2} - 1 \right)
\end{align}$$

$$\begin{align}
1 + e^{-z} &\geq \frac{2}{1 + \sqrt{1 - 4/\left\vert w \right\vert}} \\
e^{-z} &\geq \frac{2}{1 + \sqrt{1 - 4/\left\vert w \right\vert}} - 1 \\
e^{-z} &\geq \frac{\left\vert w \right\vert \left(1 - \sqrt{1 - 4/\left\vert w \right\vert}\right)}{2} - 1 \\
-z &\geq \ln \left(\frac{\left\vert w \right\vert \left(1 - \sqrt{1 - 4/\left\vert w \right\vert}\right)}{2} - 1 \right) \\
z &\leq - \ln \left(\frac{\left\vert w \right\vert \left(1 - \sqrt{1 - 4/\left\vert w \right\vert}\right)}{2} - 1 \right)
\end{align}$$

$$- \ln\left(\frac{\left\vert w \right\vert \left(1 + \sqrt{1 - 4/\left\vert w \right\vert}\right)}{2} - 1 \right) \leq z \leq - \ln \left(\frac{\left\vert w \right\vert \left(1 - \sqrt{1 - 4/\left\vert w \right\vert}\right)}{2} - 1 \right)$$

- Show numerically that the above expression bounding the width of the range is greatest at $\left\vert w \right\vert \approx 6.9$, where it takes a value $\approx 0.45$. And so even given that everything lines up just perfectly, we still have a fairly narrow range of input activations which can avoid the vanishing gradient problem.

$$f(w) = \frac{2}{w} \ln\left(\frac{w \left(1+\sqrt{1-4/w}\right)}{2}-1\right).$$


```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
def f(w):
    return 2/w*np.log(w*(1+np.sqrt(1 - 4/w))/2 - 1)
```


```python
x = np.arange(4.0,100.0,0.1)
y = f(x)
plt.plot(x,y)

np.max(y)
x[np.argmax(y)]
```

- **Idetity neuron**: Consider a neuron with a single input, $x$, a corresponding weight, $w_1$, a bias $b$, and a weight $w_2$ on the output. Show that by choosing the weights and bias appropriately, we can ensure $w_2 \sigma(w_1 x + b) \approx x$ for $x\in [0,1]$. Such a neuron can thus be used as a kind of identify neuron, that is, a neuron whose output is the same (up to rescaling by a weight factor) as its input. *Hint: It helps to rewrite $x = 1/2 + \Delta$, to assume $w_1$ is small, and to use a Taylor series expansion in $w_1 \Delta$.*