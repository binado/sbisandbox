import torch
from torch.distributions import MultivariateNormal
import pyro
import pyro.distributions as dist

from ..toymodel import ToyModel


class GaussianLinearToyModel(ToyModel):
    def __init__(self, n: int = 10, cov: float = 0.1):
        super().__init__(theta_event_shape=(n,), x_event_shape=(n,))
        self.prior_loc = torch.zeros(n)
        self.covariance_matrix = cov * torch.eye(n)

    def _pyro_model(self):
        theta = pyro.sample("theta", self.prior)
        return pyro.sample("x", dist.MultivariateNormal(theta, self.covariance_matrix))

    @property
    def prior(self):
        return MultivariateNormal(
            self.prior_loc, covariance_matrix=self.covariance_matrix
        )
