import torch
from torch import Tensor
import pyro
import pyro.distributions as dist

from ..benchmark import Benchmark, UniformPriorMixin


class SLCPBenchmark(UniformPriorMixin, Benchmark):
    r"""SLCP benchmark task.

    The parameters $\boldsymbol{\theta} \in \mathbb{R}^5$ are sampled independently from a uniform distribution,

    $$ \theta_i \sim \mathcal{U}([-3, 3]),$$

    for $i \in \{1, \ldots, 5\}$.

    The data $\boldsymbol{x} \in \mathbb{R}^8$ are generated as follows:

    $$ (\boldsymbol{x}_1, \boldsymbol{x}_2, \boldsymbol{x}_3, \boldsymbol{x}_4) \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}),$$

    where

    $$ \boldsymbol{\mu} = \begin{pmatrix} \theta_1 \\ \theta_2 \end{pmatrix}$$

    and

    $$ \boldsymbol{\Sigma} =
    \begin{pmatrix}
        \theta_3^4 &  \theta_3^2 \theta_4^2 \tanh \theta_5 \\
         \theta_3^2 \theta_4^2 \tanh \theta_5  & \theta_4^4 \\
    \end{pmatrix}.
    $$
    """

    def __init__(self, eps=1e-5) -> None:
        super().__init__(theta_event_shape=(5,), x_event_shape=(4, 2))
        self.event_batch_size = self.x_event_shape[0]
        self._eps = eps

    @property
    def low(self) -> torch.Tensor:
        return -3 * torch.ones(self.params_dimensionality)

    @property
    def high(self) -> torch.Tensor:
        return 3 * torch.ones(self.params_dimensionality)

    def _pyro_model(self):
        theta = pyro.sample("theta", self.prior)
        loc = theta[..., :2]
        s1 = theta[..., 2] ** 2
        s2 = theta[..., 3] ** 2
        rho = torch.tanh(theta[..., 4])
        cov = torch.empty((theta.shape[0], 2, 2))
        cov[..., 0, 0] = s1**2 + self._eps
        cov[..., 1, 1] = s2**2 + self._eps
        cov[..., 0, 1] = rho * s1 * s2
        cov[..., 1, 0] = rho * s1 * s2
        with pyro.plate("event dim", self.event_batch_size, dim=-1):
            _dist = dist.MultivariateNormal(loc.unsqueeze(1), cov.unsqueeze(1))
            return pyro.sample("x", _dist)

    def simulator(self, theta: Tensor) -> Tensor:
        @pyro.poutine.condition(data={"theta": theta})
        def _simulator():
            with pyro.plate("data", theta.shape[0], dim=-2):
                return self._pyro_model()

        return _simulator()
