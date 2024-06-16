import pyro

from ..toymodel import ToyModel
from ..samplers import PyroMCMCPosterior

from .benchmark import Benchmark


class MCMCBenchmark(Benchmark):
    def __init__(
        self, toy_model: ToyModel, seed: int, method: str = "NUTS", **kwargs
    ) -> None:
        pyro.set_rng_seed(seed)
        super().__init__(toy_model, seed)
        self.posterior = PyroMCMCPosterior(toy_model, method, **kwargs)
