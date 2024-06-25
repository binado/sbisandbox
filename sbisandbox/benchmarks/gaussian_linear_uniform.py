import torch
from torch import Tensor

import pyro
import pyro.distributions as dist

from ..types import Shape
from ..benchmark import Benchmark, UniformPriorMixin


class GaussianLinearUniformBenchmark(UniformPriorMixin, Benchmark):
    r"""Gaussian linear benchmark task.

    The parameters $\boldsymbol{\theta} \in \mathbb{R}^n$ are sampled independently from a uniform distribution,

    $$ \theta_i \sim \mathcal{U}([-1, 1]),$$

    for $i \in \{1, \ldots, n\}$. The data $\boldsymbol{x} \in \mathbb{R}^n$ are generated as follows:

    $$ \boldsymbol{x} \sim \mathcal{N}(\mu=\boldsymbol{\theta}, \Sigma_2=\sigma \boldsymbol{I}_n) $$

    The posterior is analytical and equal to the likelihood, i.e

    $$
    \log p(\boldsymbol{\theta} | \boldsymbol{x}) \propto -\frac{1}{2\sigma^2}(\boldsymbol{x} - \boldsymbol{\theta})^T(\boldsymbol{x} - \boldsymbol{\theta})
    $$
    """

    def __init__(self, n: int = 10, cov: float = 0.1):
        super().__init__(theta_event_shape=(n,), x_event_shape=(n,))
        self.prior_loc = torch.zeros(n)
        self.covariance_matrix = cov * torch.eye(n)

    @property
    def low(self):
        return -1 * torch.ones(self.params_dimensionality)

    @property
    def high(self):
        return torch.ones(self.params_dimensionality)

    def _pyro_model(self):
        theta = pyro.sample("theta", self.prior)
        return pyro.sample("x", dist.MultivariateNormal(theta, self.covariance_matrix))

    def loc(self, theta: torch.Tensor) -> torch.Tensor:
        return theta

    def get_posterior_samples(self, shape: Shape, x: Tensor) -> Tensor:
        return (
            dist.MultivariateNormal(loc=x, covariance_matrix=self.covariance_matrix)
            .expand(shape)
            .sample()
        )
