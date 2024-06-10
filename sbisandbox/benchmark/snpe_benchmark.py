from typing import Optional
import logging

from sbi.inference import SNPE
from sbi.utils import RestrictedPrior, get_density_thresholder

from ..toymodel import ToyModel
from .benchmark import Benchmark

logger = logging.getLogger(__name__)


class SNPEBenchmark(Benchmark):
    def __init__(
        self, toy_model: ToyModel, seed: int, density_estimator: Optional[str] = None
    ) -> None:
        super().__init__(toy_model, seed)
        self._inference = SNPE(prior=self.prior, density_estimator=density_estimator)

    def train(
        self,
        num_simulations: int,
        num_rounds: int = 1,
        truncate_at: Optional[float] = None,
    ):
        num_simulations_per_round = num_simulations // num_rounds
        proposal = self.prior
        for i in range(num_rounds):
            logger.info("Starting training round %s", i + 1)
            theta, x = self.simulate(num_simulations_per_round, proposal=proposal)
            density_estimator = self._instance.append_simulations(
                theta, x, proposal=proposal
            ).train()
            posterior = self._inference.build_posterior(
                density_estimator
            ).set_default_x(self.toy_model.x0)

            if truncate_at is not None:
                accept_reject_fn = get_density_thresholder(
                    posterior, quantile=truncate_at
                )
                proposal = RestrictedPrior(
                    self.prior, accept_reject_fn, sample_with="rejection"
                )
            else:
                proposal = posterior

        return posterior, density_estimator
