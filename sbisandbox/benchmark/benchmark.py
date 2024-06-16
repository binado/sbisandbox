from abc import ABC
from time import time
from typing import Sequence

from torch import Tensor
from sbi.inference import simulate_for_sbi

from ..toymodel import ToyModel
from ..utils import validate_model


class Benchmark(ABC):
    def __init__(self, toy_model: ToyModel, seed: int) -> None:
        self.toy_model = toy_model
        self.prior, self.simulator = validate_model(
            toy_model.prior, toy_model.simulator
        )
        self.seed = seed
        self.posterior = None
        self._sampling_time = None
        self._sampling_terminated = False

    @property
    def x0(self):
        return self.toy_model.x0

    @property
    def sampling_time(self):
        assert self._sampling_terminated, "Sampling has not yet terminated"
        return self._sampling_time

    def simulate(
        self, num_simulations: int, proposal=None, **kwargs
    ) -> Sequence[Tensor]:
        _proposal = proposal or self.prior
        theta, x = simulate_for_sbi(
            self.simulator,
            proposal=_proposal,
            num_simulations=num_simulations,
            seed=self.seed,
            **kwargs,
        )
        return theta, x

    def sample(self, num_samples, x: Tensor = None, **kwargs):
        self._sampling_terminated = False
        self._sampling_time = time()
        samples = self.posterior.sample((num_samples,), x=x, **kwargs)
        self._sampling_terminated = True
        self._sampling_time = time() - self._sampling_time
        return samples
