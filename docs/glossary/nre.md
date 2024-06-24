# Neural Ratio Estimation (NRE)

## Introduction

As we have seen, the output of prior + simulator is the array of pairs $(\boldsymbol{x}_i, \boldsymbol{\theta}_i)$ is drawn from the joint distribution

$$
    (\boldsymbol{x}_i, \boldsymbol{\theta}_i) \sim p(\boldsymbol{x}, \boldsymbol{\theta}) = p(\boldsymbol{x} | \boldsymbol{\theta})p(\boldsymbol{\theta})
$$

We now consider the shuffled pairs $(\boldsymbol{x}_i, \boldsymbol{\theta}_j)$, where $\boldsymbol{x}_i$ is the output of the forward-modeled input $\boldsymbol{\theta}_i, \, i \neq j$. These pairs are sampled from the product distribution

$$
    (\boldsymbol{x}_i, \boldsymbol{\theta}_j) \sim p(\boldsymbol{x}) p(\boldsymbol{\theta})
$$

The idea of NRE is to train a classifier to learn the ratio

$$
    r(\boldsymbol{x}, \boldsymbol{\theta}) \equiv \frac{p(\boldsymbol{x}, \boldsymbol{\theta})}{p(\boldsymbol{x})p(\boldsymbol{\theta})} = \frac{p(\boldsymbol{x} | \boldsymbol{\theta})}{p(\boldsymbol{x})},
$$

which is equal to the likelihood-to-evidence ratio. The application of Bayes' theorem makes the connection between $r(\boldsymbol{x}, \boldsymbol{\theta})$ and the Bayesian inverse problem:

$$
    r(\boldsymbol{x}, \boldsymbol{\theta}) = \frac{p(\boldsymbol{x}, \boldsymbol{\theta})}{p(\boldsymbol{x})} = \frac{p(\boldsymbol{\theta} | \boldsymbol{x})}{p(\boldsymbol{\theta})}.
$$

In other words, $r(\boldsymbol{x}, \boldsymbol{\theta})$ equals the posterior-to-prior ratio. Therefore, one can get samples from the posterior distribution of $\boldsymbol{\theta}$ from the approximate knowledge of $r(\boldsymbol{x}, \boldsymbol{\theta})$ and prior samples from $\boldsymbol{\theta}$.

More specifically, the binary classifier $d_{\boldsymbol{\phi}} (\boldsymbol{x}, \boldsymbol{\theta})$ with learnable parameters $\boldsymbol{\phi}$ is trained to distinguish the $(\boldsymbol{x}_i, \boldsymbol{\theta}_i)$ pairs sampled from the joint distribution from their shuffled counterparts. We label pairs with a variable $y$, such that $y=1$ refers to joint pairs, and $y=0$ to shuffled pairs. The classifier is trained to approximate

\begin{equation*}
\begin{split}
    d_{\boldsymbol{\phi}} (\boldsymbol{x}, \boldsymbol{\theta}) &\approx p(y=1 | \boldsymbol{x}, \boldsymbol{\theta})\\
    &= \frac{p(\boldsymbol{x}, \boldsymbol{\theta} | y = 1) p(y = 1)}{p(\boldsymbol{x}, \boldsymbol{\theta} | y = 0) p(y = 0) + p(\boldsymbol{x}, \boldsymbol{\theta} | y = 1) p(y = 1)}\\
    &= \frac{p(\boldsymbol{x}, \boldsymbol{\theta})}{p(\boldsymbol{x})\boldsymbol{\theta} + p(\boldsymbol{x}, \boldsymbol{\theta})}\\
    &= \frac{r(\boldsymbol{x}, \boldsymbol{\theta)}}{1 + r(\boldsymbol{x}, \boldsymbol{\theta)}},
\end{split}
\end{equation*}

where we used $p(y=0)=p(y=1)=0.5$.

The classifier learns the parameters $\boldsymbol{\phi}$ by minimizing the binary-cross entropy, defined as

$$
    L(d_{\boldsymbol{\phi}}) = - \int d\boldsymbol{\theta} \int d\boldsymbol{x} p(\boldsymbol{x}, \boldsymbol{\theta})\log d_{\boldsymbol{\phi}}(\boldsymbol{x}, \boldsymbol{\theta}) - p(\boldsymbol{x})\boldsymbol{\theta}\log(1-d_{\boldsymbol{\phi}}(\boldsymbol{x}, \boldsymbol{\theta}))
$$

## References

[1]: Hermans, Joeri, Volodimir Begy, and Gilles Louppe. "Likelihood-free mcmc with amortized approximate ratio estimators." International conference on machine learning. PMLR, 2020.

[2]: Miller, Benjamin K., et al. "Truncated marginal neural ratio estimation." Advances in Neural Information Processing Systems 34 (2021): 129-143.

[3]: Anau Montel, Noemi, James Alvey, and Christoph Weniger. "Scalable inference with autoregressive neural ratio estimation." Monthly Notices of the Royal Astronomical Society 530.4 (2024): 4107-4124.
