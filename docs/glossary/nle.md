# Neural Likelihood Estimation (NLE)

## Introduction

Neural Posterior Estimation (NPE) consists of training a neural density estimator with a simulated dataset to directly approximate the likelihood $p(\boldsymbol{x} | \boldsymbol{\theta})$.

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

## References

[1]: Papamakarios, George, David Sterratt, and Iain Murray. "Sequential neural likelihood: Fast likelihood-free inference with autoregressive flows." The 22nd international conference on artificial intelligence and statistics. PMLR, 2019.
