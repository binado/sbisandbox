import math

import pyro
import pyro.distributions as dist
import torch
from torch.distributions import Normal
import swyft

from ..toymodel import ToyModel, UniformPriorMixin


class TwoMoonsToyModel(UniformPriorMixin, ToyModel):
    def __init__(self):
        super().__init__(theta_event_shape=(2,), x_event_shape=(2,))
        self.r_dist = Normal(0.1, 0.01)
        self.offset = torch.tensor([0.25, 0.0])

    @property
    def x0(self):
        return torch.tensor([0, 0])

    @property
    def low(self):
        return -1 * torch.ones(self.theta_event_shape[0])

    @property
    def high(self):
        return torch.ones(self.theta_event_shape[0])

    def _pyro_model(self) -> torch.Tensor:
        offset = pyro.param("offset", self.offset)
        theta_1 = pyro.sample("theta_1", dist.Uniform(low=-1, high=1))
        theta_2 = pyro.sample("theta_2", dist.Uniform(low=-1, high=1))
        a = pyro.sample("a", dist.Uniform(low=-0.5 * math.pi, high=0.5 * math.pi))
        r = pyro.sample("r", dist.Normal(loc=0.1, scale=0.01))
        p = torch.stack((r * torch.cos(a), r * torch.sin(a)), dim=1) + offset
        abssum = torch.abs(theta_1 + theta_2)
        diff = theta_1 - theta_2
        x = pyro.deterministic(
            "x", p - torch.stack((abssum, diff), dim=1) / math.sqrt(2)
        )
        return x

    # def simulator(self, params: torch.Tensor) -> torch.Tensor:
    #     nsamples = params.shape[0]
    #     a = -0.5 + math.pi * torch.rand(nsamples)
    #     r = self.r_dist.sample((nsamples,))
    #     p = torch.stack((r * torch.cos(a), r * torch.sin(a)), dim=1) + self.offset
    #     abssum = torch.abs(torch.sum(params, 1))
    #     diff = params[:, 0] - params[:, 1]
    #     return p - torch.stack((abssum, diff), dim=1) / math.sqrt(2)
    # def simulator(self, params: torch.Tensor) -> torch.Tensor:
    #     conditioned_model = pyro.poutine.condition(
    #         self._pyro_model,
    #         data={"theta_1": params[..., 0], "theta_2": params[..., 1]},
    #     )
    #     return conditioned_model(params.shape[0])

    def loglike(self, params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        abssum = torch.abs(torch.sum(params, 1))
        diff = params[:, 0] - params[:, 1]
        # (-|p1 + p2|, p1 - p2) / sqrt(2)
        rotated_params = torch.stack((abssum, diff), dim=1) / math.sqrt(2)
        p = x + rotated_params - self.offset
        # r is normally distributed as N(0.1, 0.01)
        r = torch.sum(p**2)
        return self.r_dist.log_prob(r)


class TwoMoonsSwyftSimulator(swyft.Simulator):
    def __init__(self):
        super().__init__()
        self._simulator = TwoMoonsToyModel()

    def get_params(self):
        return self._simulator.prior.sample()

    def get_a(self):
        return -0.5 + math.pi * torch.rand()

    def get_r(self):
        return torch.normal(0.1, 0.01)

    def get_p(self, a, r):
        return torch.stack((r * torch.cos(a) + 0.25, r * torch.sin(a)), dim=1)

    def get_x(self, p, params):
        abssum = torch.abs(torch.sum(params, 1))
        diff = params[:, 0] - params[:, 1]
        return p - torch.stack((abssum, diff), dim=1) / math.sqrt(2)

    def build(self, graph):
        params = graph.node("params", self.get_params)
        a = graph.node("a", self.get_a)
        r = graph.node("r", self.get_r)
        p = graph.node("p", self.get_p, a, r)
        x = graph.node("x", self.get_x, p, params)  # noqa: F841
