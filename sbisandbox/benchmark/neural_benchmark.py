from abc import ABC, abstractmethod
from time import time
from typing import Type, Optional
import logging
from warnings import warn

from torch import Tensor
from sbi.inference import NeuralInference


from sbisandbox.toymodel import ToyModel
from .benchmark import Benchmark

logger = logging.getLogger(__name__)


class NeuralBenchmark(Benchmark, ABC):
    inference_cls: Type[NeuralInference] = None
    inference_algorithm: str = None

    def __init__(self, toy_model: ToyModel, seed: int, **kwargs) -> None:
        super().__init__(toy_model, seed)
        self._training_time = None
        self._training_clock_ended = False

        self._instance_kwargs = kwargs
        assert self.inference_cls is not None, "Must set inference_cls attribute"
        self.density_estimator = None
        self.reset()

    def reset(self):
        self._inference = self.inference_cls(prior=self.prior, **self._instance_kwargs)

    @property
    def training_time(self):
        assert (
            self._training_clock_ended
        ), ".train() method has not yet successfully finished"
        return self._training_time

    @abstractmethod
    def _train_round(
        self,
        proposal,
        num_simulations: int,
        x_0: Optional[Tensor] = None,
        posterior_kwargs: Optional[dict] = None,
        training_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        raise NotImplementedError

    def train(
        self,
        num_simulations: int,
        num_rounds: int = 1,
        x_0: Optional[Tensor] = None,
        posterior_kwargs: Optional[dict] = None,
        training_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        if num_rounds > 1 and x_0 is None:
            warn("Reference observation x_0 not set")
        self._start_training_clock()
        num_simulations_per_round = num_simulations // num_rounds
        proposal = self.prior
        for i in range(num_rounds):
            logger.info(
                "Starting training round %s for %s", i + 1, self.inference_algorithm
            )
            new_proposal, density_estimator = self._train_round(
                proposal,
                num_simulations_per_round,
                x_0,
                posterior_kwargs,
                training_kwargs,
                **kwargs,
            )
            proposal = new_proposal

        self._end_training_clock()
        self.posterior = proposal
        self.density_estimator = density_estimator
        return self.posterior, self.density_estimator

    def _start_training_clock(self):
        self._training_clock_ended = False
        self._training_time = time()

    def _end_training_clock(self):
        self._training_clock_ended = True
        self._training_time = time() - self._training_time
