import torch
from torch import Tensor
import pyro
import pyro.distributions as dist

from ..types import Shape
from ..toymodel import ToyModel, UniformPriorMixin


class GaussianMixtureToyModel(UniformPriorMixin, ToyModel):
    def __init__(self, n: int = 2, cov_high: float = 1.0, cov_low: float = 0.01):
        super().__init__(theta_event_shape=(n,), x_event_shape=(n,))
        self.prior_loc = torch.zeros(n)
        self.covariance_matrix = torch.eye(n)
        self.precision_matrix = torch.eye(n)
        self.cov_high = torch.tensor(cov_high)
        self.cov_low = torch.tensor(cov_low)

    @property
    def low(self):
        return -10 * torch.ones(self.theta_event_shape[0])

    @property
    def high(self):
        return 10 * torch.ones(self.theta_event_shape[0])

    def _pyro_model(self):
        theta = pyro.sample("theta", self.prior)
        u = pyro.sample(
            "u", dist.Categorical(0.5 * torch.ones(2)), infer={"enumerate": "parallel"}
        )
        # cov = pyro.deterministic("cov", torch.where(u > 0.5, self.cov_high, self.cov_low))
        cov = torch.where(u > 0.5, self.cov_high, self.cov_low)
        return pyro.sample(
            "x", dist.MultivariateNormal(theta, cov * self.covariance_matrix)
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
