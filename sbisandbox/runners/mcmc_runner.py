import pyro

from ..benchmark import Benchmark
from ..samplers import PyroMCMCPosterior

from .runner import Runner


class MCMCRunner(Runner):
    def __init__(
        self, toy_model: Benchmark, seed: int, method: str = "NUTS", **kwargs
    ) -> None:
        pyro.set_rng_seed(seed)
        super().__init__(toy_model, seed)
        self.posterior = PyroMCMCPosterior(toy_model, method, **kwargs)
