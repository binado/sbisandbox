from typing import Sequence

from torch import Tensor
from sbi.inference import simulate_for_sbi

from ..toymodel import ToyModel
from ..utils import validate_model, if_none_else


class Benchmark:
    def __init__(self, toy_model: ToyModel, seed: int) -> None:
        self.prior, self.simulator = validate_model(
            toy_model.prior, toy_model.simulator
        )
        self.seed = seed
        self.posterior = None

    @property
    def x0(self):
        return self.toy_model.x0

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

    def sample(self, num_samples, x: Tensor = None):
        _x = if_none_else(x, self.toy_model.x0)
        return self.posterior.sample((num_samples,), _x)
