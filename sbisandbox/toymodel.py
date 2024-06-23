from abc import ABC, abstractmethod
from typing import Optional

from torch import Tensor
from torch.distributions import MultivariateNormal, Distribution
import pyro
import pyro.distributions as dist
from sbi.inference import simulate_for_sbi

from .utils import validate_model
from .types import Shape


class ToyModel(ABC):
    def __init__(self, theta_event_shape: Shape, x_event_shape: Shape) -> None:
        self.theta_event_shape = theta_event_shape
        self.x_event_shape = x_event_shape
        self.labels = map(
            lambda n: f"theta_{n}", range(1, self.params_dimensionality + 1)
        )
        self._plates = []
        super().__init__()

    @property
    def params_dimensionality(self):
        return self.theta_event_shape[0]

    @property
    def data_dimensionality(self):
        return self.x_event_shape

    @property
    @abstractmethod
    def prior(self) -> Distribution:
        raise NotImplementedError

    @abstractmethod
    def _pyro_model(self) -> Tensor:
        raise NotImplementedError

    def conditioned_model(
        self, input: Tensor, name: str, plate_name: Optional[str] = "input"
    ) -> Tensor:
        @pyro.poutine.condition(data={name: input})
        def _conditioned_model() -> Tensor:
            with pyro.plate("data", input.shape[0]):
                return self._pyro_model()

        return _conditioned_model()

    def pyro_model_to_mcmc(self, x: Tensor) -> Tensor:
        return self.conditioned_model(x, name="x")

    def simulator(self, theta: Tensor) -> Tensor:
        return self.conditioned_model(theta, name="theta")

    def loglike(self, params: Tensor, x: Tensor) -> Tensor:
        raise NotImplementedError

    def log_prob(self, params: Tensor, x: Tensor):
        return self.loglike(params, x) + self.prior.log_prob(params)

    def get_observations(self, num_observations: int, seed: int, **kwargs):
        _prior, _simulator = validate_model(self.prior, self.simulator)
        theta, x = simulate_for_sbi(
            _simulator,
            proposal=_prior,
            num_simulations=num_observations,
            seed=seed,
            **kwargs,
        )
        return theta, x


class UniformPriorMixin:
    @property
    @abstractmethod
    def low(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def high(self):
        raise NotImplementedError

    def _prior_dist(self):
        return dist.Uniform(low=self.low, high=self.high).to_event(1)

    @property
    def prior(self):
        return self._prior_dist()


class MultivariateNormalMixin:
    theta_independtly_sampled = False

    @abstractmethod
    def loc(self, theta: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def covariance_matrix(self, theta: Tensor) -> Tensor:
        raise NotImplementedError

    def _dist(self, theta: Tensor) -> MultivariateNormal:
        loc = self.loc(theta)
        covariance_matrix = self.covariance_matrix(theta)
        return dist.MultivariateNormal(loc, covariance_matrix=covariance_matrix)

    def _pyro_model(self):
        theta = self._prior_model()
        _dist = self._dist(theta)
        return pyro.sample("x", _dist)

    def loglike(self, theta: Tensor, x: Tensor) -> Tensor:
        return self._dist(theta).log_prob(x)
