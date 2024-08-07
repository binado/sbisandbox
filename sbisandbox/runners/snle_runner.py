from typing import Optional

from torch import Tensor
from sbi.inference import SNLE

from ..benchmark import Benchmark
from .neural_runner import NeuralRunner


class SNLERunner(NeuralRunner):
    def __init__(
        self, benchmark: Benchmark, seed: int, density_estimator: str = "nsf"
    ) -> None:
        super().__init__(benchmark, seed, density_estimator=density_estimator)

    @property
    def inference_cls(self):
        return SNLE

    @property
    def inference_algorithm(self):
        return "SNLE"

    def _train_round(
        self,
        proposal,
        num_simulations: int,
        x_0: Optional[Tensor] = None,
        posterior_kwargs: Optional[dict] = None,
        training_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        _training_kwargs = training_kwargs or {}
        _posterior_kwargs = posterior_kwargs or {}
        theta, x = self.simulate(num_simulations, proposal=proposal, **kwargs)
        density_estimator = self._inference.append_simulations(theta, x).train(
            **_training_kwargs
        )
        posterior = self._inference.build_posterior(
            density_estimator=density_estimator, **_posterior_kwargs
        )
        new_proposal = posterior.set_default_x(x_0) if x_0 is not None else posterior
        return new_proposal, posterior, density_estimator
