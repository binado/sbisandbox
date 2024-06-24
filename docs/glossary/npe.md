# Neural Posterior Estimation (NPE)

## Introduction

Neural Posterior Estimation (NPE) consists of training a neural density estimator with a simulated dataset to directly approximate the posterior $p(\boldsymbol{\theta} | \boldsymbol{x})$.

The estimator is trained to minimize the loss function

$$
    \mathcal{L}(\boldsymbol{\phi}) = \mathbb{E}_{\boldsymbol{\theta} \sim p(\boldsymbol{\theta})} \mathbb{E}_{\boldsymbol{x} \sim p(\boldsymbol{x} | \boldsymbol{\theta})} \left[-\log q_{\boldsymbol{\phi}} (\boldsymbol{\theta} | \boldsymbol{x}) \right],
$$

where $\boldsymbol{\phi}$ is the parameter vector of the neural network. The loss function attains a minimum at $q_{\boldsymbol{\phi}} (\boldsymbol{\theta} | \boldsymbol{x}) = p(\boldsymbol{\theta} | \boldsymbol{x})$. Indeed, by writing it explicity,

$$
    \mathcal{L} = -\iint d\boldsymbol{\theta} d\boldsymbol{x}  p(\boldsymbol{\theta}) p(\boldsymbol{x} | \boldsymbol{\theta}) \log q_{\boldsymbol{\phi}}(\boldsymbol{\theta} | \boldsymbol{x}),
$$

one can apply Bayes' theorem and commute the integrals to write
$$
\begin{split}
    \mathcal{L} &= -\int d\boldsymbol{x} p(\boldsymbol{x}) \int d\boldsymbol{\theta} p(\boldsymbol{\theta} | \boldsymbol{x}) \log q_{\boldsymbol{\phi}}(\boldsymbol{\theta} | \boldsymbol{x}) \\
    &=D_{KL}\left[q_{\boldsymbol{\phi}}(\boldsymbol{\theta} | \boldsymbol{x}) \parallel p(\boldsymbol{\theta} | \boldsymbol{x}) \right] + \rm{const},
\end{split}
$$

where the first term is recognized to be the conditional relative entropy between $q_{\boldsymbol{\phi}}(\boldsymbol{\theta} | \boldsymbol{x})$ and the true posterior distribution $p(\boldsymbol{\theta} | \boldsymbol{x})$, which is zero if and only if the two measures are equal almost everywhere, and positive otherwise. The additional constant term does not depend on  $q_{\boldsymbol{\phi}}$ and equals

$$
    \mathbb{E}_{\boldsymbol{\theta} \sim p(\boldsymbol{\theta})} \mathbb{E}_{\boldsymbol{x} \sim p(\boldsymbol{x} | \boldsymbol{\theta})} \left[-\log p(\boldsymbol{\theta} | \boldsymbol{x}) \right].
$$

A common implementation of the density estimator is a [normalizing flow](./nflow.md).

## Comparison with other methods

One advantage of this approach is that one can obtain samples from the posterior directly from the density estimator itself, contrary to [NLE](./nle.md) or [NRE](./nre.md).

## References

[1]: Papamakarios, George, and Iain Murray. "Fast ε-free inference of simulation models with bayesian conditional density estimation." Advances in neural information processing systems 29 (2016).

[2]: Lueckmann, Jan-Matthis, et al. "Flexible statistical inference for mechanistic models of neural dynamics." Advances in neural information processing systems 30 (2017).

[3]: Greenberg, David, Marcel Nonnenmacher, and Jakob Macke. "Automatic posterior transformation for likelihood-free inference." International Conference on Machine Learning. PMLR, 2019.

[4]: Deistler, Michael, Pedro J. Goncalves, and Jakob H. Macke. "Truncated proposals for scalable and hassle-free simulation-based inference." Advances in Neural Information Processing Systems 35 (2022): 23135-23149.
