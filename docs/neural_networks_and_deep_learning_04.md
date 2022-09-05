---
layout: post
title: A visual proof that neural nets can compute any function
tags: [deep-learning, universality]
date: 2018-05-14 10:00:00 +0800
---

## A visual proof that neural nets can compute any function
- One of the most striking facts about neural networks is that they can compute any function at all. No matter what the function, there is guaranteed to be a neural networks so that for every possible input, $x$, the value $f(x)$ (or some close approximation) is output from the network.

### Two caveats
- First, this doesn't mean that a network can be used to _exactly_ compute any funciton. Rather, we can get an _approximation_ that is as good as we want. By increasing the number of hidden neurons we can improve the approximation.
- To make this statement more precise, suppose we're given a function $f(x)$ which we'd like to compute to within some desired accuracy $\epsilon > 0$. The guarantee is that by using enough hiidden neurons we can always find a neural network whose output $g(x)$ satisfies $\vert g(x) - f(x) \vert < \epsilon$, for all inputs $x$. In other words, the approximation will be good to within the desired accuracy for every possible input.
- The second caveat is that the class of functions which can be approximated in the way described are the _continuous_ functions. If a funciton is discontinuous, i.e., makes sudden, sharp jumps, then it won't in general be possible to approximate using a neural net. This is not surprising, since our neural networks compute continuous functions of their input. However, even if the function we'd really like to compute is discontinuous, it's often the case that a continuous approximation is good enough. If that's so, then we can use a neural network. In practice, this is not usually an important limitation.
- Summing up, a more precise statement of the universality theorem is that neural networks with a single hidden layer can be used to approximate any continuous function to any desired precision.

### Universality with one input and one output
- As the bias increases the graph moves to the left, but its shape doesn't change; as the bias decreases the graph moves to the right, but, again, its shape doesn't change. And as you decrease the weight, the curve broadens out; and as the input weight gets larger the output approaches a step function.
- It's actually quite a bit easier to work with step functions than general sigmoid functions. The reason is that in the output layer we add up contributions from all the hidden neurons. It's easy to analyze the sum of bunch of step functions, but rather more difficult to reason about what happens when you add up a bunch of sigmoid shaped curves. And so it makes things much easier to assume that our hidden neurons are outputting step functions.
- In essence, we're using our single-layer neural networks to build a lookup table for the function.

### Many input variables

#### A visual proof that neural nets can compute any function

**Problem**.

- We've seen how to use networks with two hidden layers to approximate an arbitrary function. Can you find a proof showing that it's possible with just a single hidden layer? As a hint, try working in the case of just two input variables, and showing that:
    1. it's possible to get step functions not just in the $x$ and $y$ directions, but in an arbitrary direction;
    2. by adding up many of the constrcutions from part (1) it's possible to approximate a tower function which is circular in shape, rather than rectangular;
    3. using these circular towers, it's possible to approximate an arbitrary function.


```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
```


```python
def step(plane, line, val=1, dim=101, positive=True):
    """The step function."""
    X, Y = np.meshgrid(np.arange(dim), np.arange(dim))
    line_graph = line[0]*X + line[1]*Y + line[2]
    if positive:
        plane[np.where(line_graph > 0)] += val
    else:
        plane[np.where(line_graph < 0)] += val
    return plane
```


```python
def generate_lines(center, radius, resolution=8):
    """Generate line which is tangent to circle."""
    thetas = np.linspace(0, 2*np.pi, resolution+1)[:-1]
    lines = np.zeros((resolution, 3))
    betas = thetas - np.pi/2
    lines[:,0] = np.sin(betas)
    lines[:,1] = - np.cos(betas)
    lines[:,2] = ((center[1] + radius*np.sin(thetas))*np.cos(betas)-
                  (center[0] + radius*np.cos(thetas))*np.sin(betas))
    return lines
```


```python
def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))
```


```python
center = [50,50]
radius = 10
resolution = 256
lines = generate_lines(center, radius, resolution=resolution)

dim = 101
plane = np.zeros((dim,dim))
for i in range(resolution):
    plane = step(plane, lines[i], dim=dim)
plane = plane - resolution*15/16
plane = sigmoid(plane)
```


```python
plt.imshow(plane)
```


```python
fig = plt.figure()
ax = Axes3D(fig)
X, Y = np.meshgrid(np.arange(dim), np.arange(dim))
ax.plot_surface(X, Y, plane)
plt.show()
```

### Extension beyong sigmoid neurons

**Problem**.

- Earlier in the book we met another type of neuron known as a rectified linear unit. Explain why such neurons don't satisfy the conditions just given for universality. Find a proof of universality showing that rectified linear units are universal for computation.
- Suppose we consider linear neurons, i.e., neurons with the activation function $s(z) = z$. Explain why linear neurons don't satisfy the conditions just given for universality. Show that such neurons can't be used to do universal computation.

### Fixing up the step functions
- Now we have two different approximations to $\sigma^{-1} \circ f(x)/2$. If we add up the two approximations we'll get an overall approximation to $\sigma^{-1} \circ f(x)/2$. That overall approximation will still have failures in small windows. But the problem will be much less than before. The reason is that points in a failure window for one approximation won't be in a failure window for the other. And so the approximation will be a factor roughly 2 better in those windows.
- We could do even better by adding up a large number, $M$, of overlapping approximations to the function $\sigma^{-1} \circ f(x)/M$. Provided the windows of failure are narrow enough, a point will only ever be in one window of failure. And provided we're using a large enough number $M$ of overlapping approximations, the result will be an excellent overall approximation.

### Conclusion
- The explanation for universality we've discussed is certainly not a practical prescription for how to compute using neural networks! In this, it's much like proofs of universality for NAND gates and the like. For this reason, I've focused mostly on trying to make the construction clear and easy to follow, and not on optimizing the details of the construction. However, you may find it a fun and instructive exercise to see if you can improve the construction.
- Although the result isn't directly useful in constructing networks, it's important because it takes off the table the question of whether any particular function is computable using a neural network. The answer to that question is always "yes". So the right question to ask is not whether any particular function is computable, but rather what's a _good_ way to compute the function.

