---
layout: post
title: A guide to convolution arithmetic for deep learning
tags: [deep-learning, convolution]
date: 2018-05-27 18:00:00 +0800
---

1. Explain the relationship between convolutional layers and transposed convolutional layers.
2. Provide an intuitive understanding of the relationship between input shape, kernel shape, zero padding, strides and output shape in convolutional, pooling and transposed convolutional layers.

## Convolution arithmetic

- **Relationship 6**. For any $i$, $k$, $p$ and $s$, $$o = \left\lfloor\frac{i + 2p - k}{s}\right\rfloor + 1.$$

## Pooling arithmetic

- In a neural network, pooling layers provide invariance to small translations of the input.
- **Relationship 7**. For any $i$, $k$ and $s$, $$o = \left\lfloor\frac{i - k}{s}\right\rfloor + 1.$$

## Transposed convolution arithmetic

- The need for transposed convolutions generally arises from the desire to use a transformation going in the opposite direction of a normal convolution, i.e., from something that has the shape of the output of some convolution to something that has the shape of its input while maintaining a connectivity pattern that is compatible with said convolution. For instance, one might use such a transformation as the decoding layer of a convolutional autoencoder or to project feature maps to a higher-dimensional space.
- The convolutional case is considerably more complex than the fully-connected case, which only requires to use a weight matrix whose shape has been transposed. However, since every convolution boils down to an efficient implementation of a matrix operation, the insights gained from the fully-connected case are useful in solving the convolutional case.

### Transposed convolution

- Transposed convolutions - also called _fractionally strided convolutions_ or _deconvolution_ - work by swapping the forward and backward passes of a convolution. One way to put it is to note that the kernel defines a convolution, but whether it's a direct convolution or a transposed convolution is determined by how the forward and backward passes are computed.
- The transposed convolution operation can be thought of as the gradient of _some_ convoluton with respect to its input, which is usually how transposed convolutions are implemented in practice.
- It is always possible to emulate a transposed convolution with a direct convolution. The disadvantage is that it usually involves adding many columns and rows of zeros to the input, resulting in a much less efficient implementation.
- Building on what has been introduced so far, this chapter will proceed somewhat backwards with respect to the convolution arithmetic chapter, deriving the properties of each transposed convolution by referring to the direct convolution with which it shares the kernel, and defining the equivalent direct convolution.

#### No zero padding, unit strides, transposed

- One way to understand the logic behind zero padding is to consider the connectivity pattren of the transposed convolution and use it to guide the design of the equivalent convolution. For example, the top left pixel of the input of the direct convolution only contribute to the left pixel of the output, the top right pixel is only connected to the top right output pixel, and so on.
- To maintain the same connectivity pattern in the equivalent convolution it is necessary to zero pad the input in such a way that the first (top-left) application of the kernel only touches the top-left pixel, i.e., the padding has to be equal to the size of the kernel minus one. Interestingly, this corresponds to a fully padded convolution with unit strides.

#### Zero padding, unit strides, transposed

- Knowing that the transpose of a non-padded convolution is equivalent to convolving a zero padded input, it would be reasonable to suppose that the transpose of a zero padded convolution is equivalent to convolving an input padded with _less_ zeros.
- By applying the same inductive reasoning as before, it is reasonable to expect that the equivalent convolution of the transpose of a half padded convolution is itself a half padded convolution, given that the output size of a half padded convolution is the same as its input size.

#### No zero padding, non-unit strides, transposed

- Using the same kind of inductive logic as for zero padded convolutions, one might expect that the transpose of a convolution with $s > 1$ involves an equivalent convolution with $s < 1$. As will be explained, this is a valid intuition, which is why transposed convolutions are sometimes called _fractionally strided convolutions_.

#### Zero padding, non-unit strides, transposed

- **Relationship 14**. A convolution described by $k$, $s$, and $p$ has an associated transposed convolution described by $a$, $\tilde{i'}$, $k^{\prime} = k$, $s^{\prime} = 1$, and $p^{\prime} = k - p - 1$, where $\tilde{i^{\prime}}$ is the size of the stretched input obtained by adding $s - 1$ zeros between each input unit, and $a = (i + 2p - k)$ mod $s$ represents the number of zeros added to the bottom and right edges of the input, and its output size is

$$o^{\prime} = s\left(i^{\prime} - 1\right) + a + k - 2p.$$
