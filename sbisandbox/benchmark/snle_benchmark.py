from typing import Optional
import logging

from sbi.inference import SNLE

from ..toymodel import ToyModel
from .benchmark import Benchmark

logger = logging.getLogger(__name__)


class SNLEBenchmark(Benchmark):
    def __init__(
        self, toy_model: ToyModel, seed: int, classifier: Optional[str] = None
    ) -> None:
        super().__init__(toy_model, seed)
        self._inference = SNLE(prior=self.prior, classifier=classifier)

    def train(self, num_simulations: int, num_rounds: int = 1, **kwargs):
        num_simulations_per_round = num_simulations // num_rounds
        proposal = self.prior
        for i in range(num_rounds):
            logger.info("Starting training round %s", i + 1)
            theta, x = self.simulate(num_simulations_per_round, proposal=proposal)
            density_estimator = self._inference.append_simulations(theta, x).train()
            posterior = self._inference.build_posterior(density_estimator, **kwargs)
            proposal = posterior.set_default_x(self.x_0)

        return posterior, density_estimator
