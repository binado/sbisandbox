# Normalizing flows

## Introduction

In a nutshell, a normalizing flow is based on the idea of a set of invertible and differentible transformations between two distributions. Let's say that $x$ follows a base distribution with density $p(x)$, and we consider the bijection $f: x \to z$. The transformed variable $z$ will follow a distribution with density

$$
\begin{aligned}
p'(z) &= p(f^{-1}(z)) \left| \frac{\partial f^{-1}}{\partial z} \right|\\
&= p(f^{-1}(z)) \left| \frac{\partial f}{\partial x} \right|^{-1}
\end{aligned}
$$

The above relation is easily generalized to the multidimensional case by replacing the partial derivative with the determinant of the jacobian of the transformation $\text{det} \frac{\partial f}{\partial x_i}$.

These transformation can be arbitrarily composed, e.g.

$$ f = f_n \circ \ldots f_2 \circ f_1,$$

which amounts to multiplying the jacobian determinants,

$$\text{det }\frac{\partial f}{\partial \boldsymbol{x}_n} = \prod_{i=1}^n \text{det } \frac{\partial f_i}{\partial \boldsymbol{x}_i}.$$

In the context of inference, a generative model can be generally viewed as a (possibly non-linear) transformation from a simple distribution (e.g. a *prior*) to a more complex distribution (e.g. the posterior). The idea of the method is that a well-parameterized composition of normalizing flows will approximate the *inverse* transformation.

Typically, the normalizing flows can be represented by a neural network architecture, whose input parameters $\phi$ are the subject to an optimization to maximize a suitable loss function, for instance the log-posterior, or a suitable distance metric in the case of variational inference.

## More resources

For a more formal discussion on the subject, we refer the reader to [this review paper](https://arxiv.org/abs/1908.09257).

## References

[1]: Kobyzev, Ivan, Simon JD Prince, and Marcus A. Brubaker. "Normalizing flows: An introduction and review of current methods." IEEE transactions on pattern analysis and machine intelligence 43.11 (2020): 3964-3979.
