from typing import Optional
import logging

from torch import Tensor
from sbi.inference import SNRE

from ..toymodel import ToyModel
from .neural_benchmark import NeuralBenchmark

logger = logging.getLogger(__name__)


class SNREBenchmark(NeuralBenchmark):
    inference_cls = SNRE
    inference_algorithm = "SNRE"

    def __init__(
        self, toy_model: ToyModel, seed: int, classifier: str = "resnet"
    ) -> None:
        super().__init__(toy_model, seed, classifier=classifier)

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
        theta, x = self.simulate(num_simulations, proposal=proposal)
        density_estimator = self._inference.append_simulations(theta, x).train(
            **_training_kwargs
        )
        posterior = self._inference.build_posterior(
            density_estimator=density_estimator, **_posterior_kwargs
        )
        new_proposal = posterior.set_default_x(x_0) if x_0 is not None else posterior
        return new_proposal, density_estimator

    def sample(
        self, num_samples: int, thin: int = 1, warmup_steps: int = 100
    ) -> Tensor:
        return super().sample(num_samples, thin=thin, warmup_steps=warmup_steps)
