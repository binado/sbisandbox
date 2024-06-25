from typing import Optional

from torch import Tensor
from sbi.inference import SNPE
from sbi.inference.potentials import posterior_estimator_based_potential
from sbi.inference.posteriors import ImportanceSamplingPosterior
from sbi.utils import RestrictedPrior, get_density_thresholder

from ..benchmark import Benchmark
from .neural_runner import NeuralRunner


class SNPERunner(NeuralRunner):
    def __init__(
        self, benchmark: Benchmark, seed: int, density_estimator: str = "nsf"
    ) -> None:
        super().__init__(benchmark, seed, density_estimator=density_estimator)

    @property
    def inference_cls(self):
        return SNPE

    @property
    def inference_algorithm(self):
        return "SNPE"

    def _train_round(
        self,
        proposal,
        num_simulations: int,
        x_0: Optional[Tensor] = None,
        posterior_kwargs: Optional[dict] = None,
        training_kwargs: Optional[dict] = None,
        truncate_at: Optional[float] = None,
        **kwargs,
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
        return proposal, posterior, density_estimator

    def get_importance_posterior(self, x_0: Tensor):
        assert (
            self.density_estimator is not None
        ), ".train method should be called first"
        potential_fn, transform = posterior_estimator_based_potential(
            self.density_estimator, self.prior, x_0
        )
        return ImportanceSamplingPosterior(potential_fn, self.prior, transform)
