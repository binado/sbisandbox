from abc import ABC
from time import time
from typing import Sequence

from torch import Tensor
from sbi.inference import simulate_for_sbi

from ..benchmark import Benchmark
from ..utils import validate_model


class Runner(ABC):
    def __init__(self, benchmark: Benchmark, seed: int) -> None:
        self.benchmark = benchmark
        self.prior, self.simulator = validate_model(
            benchmark.prior, benchmark.simulator
        )
        self.seed = seed
        self.posterior = None
        self._sampling_time = None
        self._sampling_terminated = False

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
