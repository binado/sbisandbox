import torch
from torch import Tensor

import pyro
import pyro.distributions as dist

from ..types import Shape
from ..toymodel import ToyModel, UniformPriorMixin


class GaussianLinearUniformToyModel(UniformPriorMixin, ToyModel):
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
