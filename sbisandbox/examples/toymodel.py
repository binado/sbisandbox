from abc import ABC, abstractmethod

import torch
from torch.distributions import Uniform, Independent, MultivariateNormal, Distribution


class ToyModel(ABC):
    def __init__(self, ndim) -> None:
        self.ndim = torch.tensor(ndim, dtype=int)
        super().__init__()

    @property
    def params_dimensionality(self):
        return self.ndim[0]

    @property
    def data_dimensionality(self):
        return self.ndim[1]

    @property
    @abstractmethod
    def prior(self):
        raise NotImplementedError

    @abstractmethod
    def simulator(self, params):
        raise NotImplementedError


class UniformPriorMixin:
    @property
    @abstractmethod
    def low(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def high(self):
        raise NotImplementedError

    @property
    def prior(self) -> Distribution:
        dist = Uniform(low=self.low, high=self.high)
        return Independent(dist, 1)


class MultivariateNormalMixin:
    @abstractmethod
    def loc(self, params: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def covariance_matrix(self, params: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def simulator(self, params: torch.Tensor) -> torch.Tensor:
        loc = self.loc(params)
        covariance_matrix = self.covariance_matrix(params)
        return MultivariateNormal(loc, covariance_matrix=covariance_matrix).sample()
