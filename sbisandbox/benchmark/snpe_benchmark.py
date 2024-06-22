from typing import Optional

from torch import Tensor
from sbi.inference import SNPE
from sbi.utils import RestrictedPrior, get_density_thresholder

from ..toymodel import ToyModel
from .neural_benchmark import NeuralBenchmark


class SNPEBenchmark(NeuralBenchmark):
    inference_cls = SNPE
    inference_algorithm = "SNPE"

    def __init__(
        self, toy_model: ToyModel, seed: int, density_estimator: str = "nsf"
    ) -> None:
        super().__init__(toy_model, seed, density_estimator=density_estimator)

    def _train_round(
        self,
        proposal,
        num_simulations: int,
        x_0: Optional[Tensor] = None,
        posterior_kwargs: Optional[dict] = None,
        training_kwargs: Optional[dict] = None,
        truncate_at: Optional[float] = None,
    ):
        truncate_proposals = truncate_at is not None
        _training_kwargs = training_kwargs or {}
        _posterior_kwargs = posterior_kwargs or {}
        theta, x = self.simulate(num_simulations, proposal=proposal)
        density_estimator = self._inference.append_simulations(
            theta, x, proposal=proposal
        ).train(force_first_round_loss=truncate_proposals, **_training_kwargs)
        posterior = self._inference.build_posterior(
            density_estimator=density_estimator, **_posterior_kwargs
        )
        if x_0 is not None:
            posterior.set_default_x(x_0)

        if truncate_proposals:
            accept_reject_fn = get_density_thresholder(posterior, quantile=truncate_at)
            proposal = RestrictedPrior(
                self.prior, accept_reject_fn, sample_with="rejection"
            )
        else:
            proposal = posterior
        return proposal, density_estimator
