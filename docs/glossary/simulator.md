# Simulator

## Introduction

The simulator is the general term for the function mapping the model parameters, $\boldsymbol{\theta}$, to the observed data, $\boldsymbol{x}$. It implicitly defines a likelihood function $p(\boldsymbol{x} | \boldsymbol{\theta})$, in the sense that it returns samples $\boldsymbol{x}$ which follow a distribution whose density is $p(\boldsymbol{x} | \boldsymbol{\theta})$. Therefore, once a prior on $\boldsymbol{\theta}$ is specified, we have all the necessary ingredients to perform Bayesian inference.

## Examples

### Gaussian linear

Consider a simple model where the output data are the parameters plus some gaussian noise:

$$ x = \theta + \sigma \varepsilon,$$

where $\varepsilon \sim \mathcal{N}(0, 1)$. This corresponds to the likelihood function

$$ \log p(x | \theta) \propto -\frac{1}{2} \left(\frac{x - \theta}{\sigma} \right)^2$$

The corresponding forward model (prior + simulator) can be implemented in the following steps:

1. $\theta \sim p(\theta)$
2. $\varepsilon \sim \mathcal{N}(0, 1)$
3. $x \leftarrow \theta + \sigma \varepsilon$
4. **return** $(\theta, x)$
