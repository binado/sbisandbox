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
    def x0(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def prior(self) -> Distribution:
        raise NotImplementedError

    @abstractmethod
    def simulator(self, params: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def loglike(self, params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def log_prob(self, params: torch.Tensor, x: torch.Tensor):
        return self.loglike(params, x) + self.prior.log_prob(params)


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

    def _dist(self, params: torch.Tensor) -> MultivariateNormal:
        loc = self.loc(params)
        covariance_matrix = self.covariance_matrix(params)
        return MultivariateNormal(loc, covariance_matrix=covariance_matrix)

    def simulator(self, params: torch.Tensor) -> torch.Tensor:
        return self._dist(params).sample()

    def loglike(self, params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self._dist(params).log_prob(x)
