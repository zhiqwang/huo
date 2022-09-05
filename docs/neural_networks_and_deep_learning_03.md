---
layout: post
title: Improving the way neural networks learn
tags: [deep-learning, backpropagation]
date: 2018-05-02 21:00:00 +0800
---

## Improving the way neural networks learn
### Introducing the cross-entropy cost function
- When we use the cross-entropy, the $\sigma^{\prime}\left(z\right)$ term get canceled out, and we no longer need worry about it being small. This cancellation is the special miracle ensured by the cross-entropy cost function. Actually, it's not really a miracle. As we'll see later, the cross-entropy was specially chosen to have just this property.

### What does the cross-entropy mean? Where does it come from
- Cross-entropy is a measure of surprise. In particular, our neurou is trying to compute the function $x \to y = y(x)$. But instead it computes the function $x \to a = a(x)$. Suppose we think of $a$ as our neuron's estimated probability that $y$ is $1$, and $1 - a$ is the estimated probability that the right value for $y$ is $0$. Then the cross-entropy measures how "surprised" we are, on average, when we learn the true value for $y$. We get low surprise if the output is what we expect, and high surprise if the output is unexpected.

### Overfitting and regularization
- If we set the hyper-parameters based on evaluations of the *test_data* it's possible we'll end up overfitting our hyper-parameters to the *test_data*. That is, we may end up finding hyper-parameters which fit particular peculiarities of the *test_data*, but where the performance of the network won't generalize to other data sets. To put it another way, you can think of the validation data as a type of training data that helps us learn good hyper-parameters. This approach to finding good hyper-parameters is sometimes known as the *hold out* method, since the *validation_data* is kept apart or *"held out"* from the *training_data*.

### Regularization
- **Weight decay or $L_2$ regularizaiton**. The idea of $L_2$ regularization is to add an extra term to the cost function, a term called the _regularization term_. $$C = C_0 + \frac{\lambda}{2n} \sum_w w^2, \tag{87}$$ where $C_0$ is the original, unregularized cost function. We need to figure out how to apply our stochastic gradient descent learning algorithm in a regularized neural network. The partial derivatives of Equation (87) gives $$\begin{align} \frac{\partial C}{\partial w} &= \frac{\partial C_0}{\partial w} + \frac{\lambda}{n} w, \tag{88} \\ \frac{\partial C}{\partial b} &= \frac{\partial C_0}{\partial b}. \tag{89} \end{align}$$ The $\partial C_0 / \partial w$ and $\partial C_0 / \partial b$ terms can be computed using backpropagation.

- Heuristically, if the cost function is unregularized, then the length of the weight vector is likely to grow, all other things being equal. Over time this can lead to the weight vector being very large indeed. This can cause the weight vector to get stuck pointing in more or less the same direction, since changes due to gradient descent only make tiny changes to the direction, when the length is long. I believe this phenomenon is making it hard for our learning algorithm to properly explore the weight space, and consequently harder to find good minima of the cost function.

### Why does regularization help reduce overfitting?
- A standard story people tell to explain what's going on is along the following lines: smaller weights are, in some sense, lower complexity, and so provide a simpler and more powerful explanation for the data, and should thus be preferred.
- One point of view is to say that in science we should go with the simpler explanation, unless compelled not to. When we find a simple model that seems to explain many data points we are tempted to shout "Eureka!" After all, it seems unlikely that a simple explanation should occur merely by coincidence. Rather, we suspect that the model must be expressing some underlying truth about the phenomenon.
- Let's see what this point of view means for neural networks. Suppose our network mostly has small weights, as will tend to happen in a regularized network. The smallness of the weights means that the behaviour of the network won't change too much if we change a few random inputs here and there. That makes it difficult for a regularized network to learn the effects of local noise in the data. Think of it as a way of making it so single pieces of evidence don't matter too much to the output of the network. Instead, a regularized network learns to respond to types of evidence which are seen often across the training set.
- By contrast, a network with large weights may change its behaviour quite a bit in response to small changes in the input. And so an unregularized network can use large weights to learn a complex model that carries a lot of information about the noise in the training data. In a nutshell, regularized networks are constrained to build relatively simple models based on patterns seen often in the training data, and are resistant to learning peculiarities of the noise in the training data. The hope is that this will force our networks to do real learning about the phenomenon at hand, and to generalize better from what they learn.
- It has been conjectured that "the dynamics of gradient descent learning in multilayer nets has a `self-regularization' effect".

### Other techniques for regularization
- Heuristically, when we dropout different sets of neurons, it's rather like we're training different neural networks. And so the dropout precedure is like averaging the effects of a very large number of different networks. The different networks will overfit in different ways, and so, hopefully, the net effect of dropout will be to reduce overfitting.
- "This technique reduces complex co-adaptations of neurons, since a neuron cannot rely on the presence of particular other neurons. It is, therefore, forced to learn more robust features that are useful in conjunction with many different random subsets of the other neurons." In other words, if we think of our network as a model which is making predictions, then we can think of dropout as a way of making sure that the model is robust to the loss of any individual piece of evidence.

### How to choose a neural network's hyper-parameters?
- Gradient descent uses a first-order approximation to the cost function as a guide to how to decrease the cost. For large $\eta$, higher-order terms in the cost function become more important, and may dominate the behaviour, causing gradient descent to break down. This is especially likely as we approach minima and quasi-minima of the cost function, since near such points the gradient becomes small, making it easier for higher-order terms to dominate behaviour.

### Variations on stochastic gradient descent
- The momentum technique modifies gradient descent in two ways that make it more similar to the physical picture. First, it introduces a notion of "velocity" for the parameters we're trying to optimize. The gradient acts to change the velocity, not (directly) the "position", in much the same way as physical forces change the velocity, and only indirectly affect position. Second, the momentum method introduces a kind of friction term, which tends to gradually reduce the velocity.
- Suppose we're using sigmoid neurons, so all activations in our network are positive. Let's consider the weights $w^{l+1}_{jk}$ input to the $j$th neuron in the $l+1$ th layer. The rules for backpropagation tell us that the associated gradient will be $a^l_k \delta^{l+1}_j$. Because the activations are positive the sign of this gradient will be the same as the sign of $\delta^{l+1}_j$. What this means is that if $\delta^{l+1}_j$ is positive then all the weights $w^{l+1}_{jk}$ will decrease during gradient descent, whild if $\delta^{l+1}_j$ is negative then all the weight $w^{l+1}_{jk}$ will increase during gradient descent. In other words, all weights to the same neuron must either increase together or decrease together.
- To give you the flavor of some of the issues, recall that sigmoid neurons stop learning when they saturate, i.e., when their output is near either $0$ and $1$. As we've seen repeatedly in this chapter, the problem is that $\sigma^{\prime}$ terms reduce the gradient, and that slows down learning. Tanh neurons suffer from a similar problem when they saturate. By contrast, increasing the weighted input to a rectified linear unit will never cause it to saturate, and so there is no corresponding learning slowdown. On the other hand, when the weighted input to a rectified linear unit is negative, the gradient vanishes, and so the neuron stops learning entirely. These are just two of the many issues that make it non-trivial to understand when and why rectified linear units perform better than sigmoid or tanh neurons.

**Exercise**.

- It's tempting to use gradient descent to try to learn good values for hyper-parameters such as $\lambda$ and $\eta$. Can you think of an obstacle to using gradient descent to determine $\lambda$? Can you think of an obstacle to using gradient descent to determine $\eta$?

**Exercise**.

- What would go wrong if we used $\mu > 1$ in the momentum technique?
- What would go wrong if we used $\mu < 0$ in the momentum technique?

