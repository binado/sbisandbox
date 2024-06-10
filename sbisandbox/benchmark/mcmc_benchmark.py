from sbisandbox.toymodel import ToyModel
from ..samplers import get_sampler

from .benchmark import Benchmark


class MCMCBenchmark(Benchmark):
    def __init__(
        self, toy_model: ToyModel, seed: int, method: str = "nuts_pyro", **kwargs
    ) -> None:
        super().__init__(toy_model, seed)
        self.posterior = get_sampler(toy_model, method, **kwargs)
