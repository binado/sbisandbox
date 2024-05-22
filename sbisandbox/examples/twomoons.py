import math

import torch
import swyft

from .toymodel import ToyModel, UniformPriorMixin

class TwoMoonsSimulator(UniformPriorMixin, ToyModel):
    def __init__(self):
        self.ndim = 2
        super(ToyModel).__init__((self.ndim, self.ndim))

    @property
    def low(self):
        return -1 * torch.ones(self.params_dimensionality)
    
    @property
    def high(self):
        return torch.ones(self.params_dimensionality)

    def simulator(self, params: torch.Tensor) -> torch.Tensor:
        nsamples = params.shape[0]
        a = -0.5 + math.pi * torch.rand(nsamples)
        r = 0.1 + 0.01 * torch.randn(nsamples)
        p =  torch.stack((r * torch.cos(a) + 0.25, r * torch.sin(a)), dim=1)
        abssum = torch.abs(torch.sum(params, 1))
        diff = params[:, 0] - params[:, 1]
        return p - torch.stack((abssum, diff), dim=1) / math.sqrt(2)


class TwoMoonsSwyftSimulator(swyft.Simulator):
    def __init__(self):
        super().__init__()
        self._simulator = TwoMoonsSimulator()

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
        params = graph.node('params', self.get_params)
        a = graph.node('a', self.get_a)
        r = graph.node('r', self.get_r)
        p = graph.node('p', self.get_p, a, r)
        x = graph.node('x', self.get_x, p, params)
