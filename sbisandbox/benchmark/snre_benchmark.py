from typing import Optional
import logging

from sbi.inference import SNRE

from ..toymodel import ToyModel
from .benchmark import Benchmark

logger = logging.getLogger(__name__)


class SNREBenchmark(Benchmark):
    def __init__(
        self, toy_model: ToyModel, seed: int, classifier: Optional[str] = None
    ) -> None:
        super().__init__(toy_model, seed)
        self._inference = SNRE(prior=self.prior, classifier=classifier)

    def train(self, num_simulations: int, num_rounds: int = 1, **kwargs):
        num_simulations_per_round = num_simulations // num_rounds
        proposal = self.prior
        for i in range(num_rounds):
            logger.info("Starting training round %s", i + 1)
            theta, x = self.simulate(num_simulations_per_round, proposal=proposal)
            ratio_estimator = self._inference.append_simulations(theta, x).train()
            posterior = self._inference.build_posterior(
                ratio_estimator=ratio_estimator, **kwargs
            )
            proposal = posterior.set_default_x(self.x_0)

        return posterior, ratio_estimator
