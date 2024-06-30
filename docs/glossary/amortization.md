# Amortized x sequential algorithms

## Amortization

One advantage of neural SBI algorithms is that the neural network is an *amortized* estimator of the posterior. That is, it learns $p(\boldsymbol{\theta} | \boldsymbol{x})$ for *any* value of $\boldsymbol{x}$, in contrast with traditional sampling methods, were the inference pipeline must be rerun with the new data.

Amortization also makes it more viable to run expected coverage tests, which are a useful diagnostic tool to identify overconfident/underconfident estimators (see [2]).

## Sequential variants

There are may be situations in which we do want to learn the posterior for only a particular observation $\boldsymbol{x}_0$. In that case, there are strategies to employ to reduce the simulation budget required for learning the posterior in the neighbourhood of $\boldsymbol{x}_0$.

The main idea is to construct a proposal distribution $\tilde{p}(\boldsymbol{\theta})$, not necessarily the prior $p(\boldsymbol{\theta})$, that generates samples whose simulator output is close to $\boldsymbol{x}_0$. In practice, the training of the neural network is performed over several rounds; the outpout estimator at the end of each round being chosen as the proposal distribution for the next.

## References

[1]: Papamakarios, George, David Sterratt, and Iain Murray. "Sequential neural likelihood: Fast likelihood-free inference with autoregressive flows." The 22nd international conference on artificial intelligence and statistics. PMLR, 2019.

[2]: Hermans, Joeri, et al. "A trust crisis in simulation-based inference? your posterior approximations can be unfaithful." arXiv preprint arXiv:2110.06581 (2021).
