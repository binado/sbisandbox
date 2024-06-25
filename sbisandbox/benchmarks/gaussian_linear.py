import torch
from torch import Tensor
import pyro
import pyro.distributions as dist

from ..types import Shape
from ..benchmark import Benchmark


class GaussianLinearBenchmark(Benchmark):
    r"""Gaussian linear benchmark task.

    The parameters $\boldsymbol{\theta} \in \mathbb{R}^n$ are sampled from

    $$ \boldsymbol{\theta} \sim \mathcal{N}(\mu=\boldsymbol{0}_n, \Sigma_1=\sigma \boldsymbol{I}_n),$$

    where $\sigma > 0$, $\boldsymbol{I}_n$ is the n x n identity matrix and $\boldsymbol{0}_n = \begin{pmatrix} 0 & \ldots & 0 \end{pmatrix}^T \in \mathbb{R}^n$.

    The data $\boldsymbol{x} \in \mathbb{R}^n$ are generated as follows:

    $$ \boldsymbol{x} \sim \mathcal{N}(\mu=\boldsymbol{\theta}, \Sigma_2=\sigma \boldsymbol{I}_n) $$
    """

    def __init__(self, n: int = 10, cov: float = 0.1):
        super().__init__(theta_event_shape=(n,), x_event_shape=(n,))
        self.prior_loc = torch.zeros(n)
        self.covariance_matrix = cov * torch.eye(n)
        self.precision_matrix = torch.eye(n) / cov

    def _pyro_model(self):
        theta = pyro.sample("theta", self.prior)
        return pyro.sample("x", dist.MultivariateNormal(theta, self.covariance_matrix))

    @property
    def prior(self):
        return dist.MultivariateNormal(
            self.prior_loc, covariance_matrix=self.covariance_matrix
        )

    def get_posterior_samples(self, shape: Shape, x: Tensor) -> Tensor:
        covariance_matrix = 0.5 * self.covariance_matrix
        loc = torch.matmul(
            torch.matmul(covariance_matrix, self.precision_matrix), x.squeeze()
        )
        return (
            dist.MultivariateNormal(loc=loc, covariance_matrix=covariance_matrix)
            .expand(shape)
            .sample()
        )
