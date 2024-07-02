import torch
from torch import Tensor
import pyro
import pyro.distributions as dist

from ..types import Shape
from ..benchmark import Benchmark, UniformPriorMixin


class GaussianMixtureBenchmark(UniformPriorMixin, Benchmark):
    r"""Gaussian mixture benchmark task.

    The parameters $\boldsymbol{\theta} \in \mathbb{R}^n$ are sampled independently from a uniform distribution,

    $$ \theta_i \sim \mathcal{U}([-1, 1]),$$

    for $i \in \{1, \ldots, n\}$.

    The data $\boldsymbol{x} \in \mathbb{R}^n$ are generated as follows:

    $$ \boldsymbol{x} \sim 0.5 \mathcal{N}(\mu=\boldsymbol{\theta}, \sigma_1 \boldsymbol{I}_n) + 0.5 \mathcal{N}(\mu=\boldsymbol{\theta}, \sigma_2 \boldsymbol{I}_n),$$

    where $\sigma_1 \gg \sigma_2 > 0$.
    """

    def __init__(self, n: int = 2, cov_high: float = 1.0, cov_low: float = 0.1):
        super().__init__(theta_event_shape=(n,), x_event_shape=(n,))
        self.prior_loc = torch.zeros(n)
        self.covariance_matrix = torch.eye(n)
        self.precision_matrix = torch.eye(n)
        self.cov_high = cov_high * torch.ones(1)
        self.cov_low = cov_low * torch.ones(1)

    @property
    def low(self):
        return -10 * torch.ones(self.theta_event_shape[0])

    @property
    def high(self):
        return 10 * torch.ones(self.theta_event_shape[0])

    def _pyro_model(self):
        theta = pyro.sample("theta", self.prior)
        u = pyro.sample(
            "u",
            dist.Categorical(probs=0.5 * torch.ones(2)),
            infer={"enumerate": "parallel"},
        )
        shape = theta.shape[:-1]  # Batch shape
        # When doing MCMC, u will be 2d because of enumerate parallel
        # There's gotta be a cleaner fix to this
        cov = torch.where(
            u > 0.5,
            self.cov_high,
            self.cov_low,
        ).view(shape + (u.shape if len(u.shape) > 1 else 2 * (1,)))
        covariance_matrix = cov * self.covariance_matrix.expand(
            shape + 2 * self.x_event_shape
        )
        return pyro.sample(
            "x", dist.MultivariateNormal(theta, covariance_matrix=covariance_matrix)
        )

    def get_posterior_samples(self, shape: Shape, x: Tensor) -> Tensor:
        u = dist.Uniform(low=0, high=1).sample(shape).unsqueeze(-1)
        cov = torch.where(
            u > 0.5,
            self.cov_high.expand(shape + (1,)),
            self.cov_low.expand(shape + (1,)),
        ).unsqueeze(-1)
        covariance_matrix = cov * self.covariance_matrix.expand(
            shape + 2 * self.x_event_shape
        )
        return dist.MultivariateNormal(
            loc=x.expand(shape + self.x_event_shape),
            covariance_matrix=covariance_matrix,
        ).sample()
