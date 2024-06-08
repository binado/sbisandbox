from typing import Iterable, Optional
from multiprocessing import Pool
import logging

from sbi.inference import NeuralInference
from torch import Tensor

from .utils import get_type_from_iterable

logger = logging.getLogger(__name__)


def _train(algo: NeuralInference):
    return algo.train()


class InferencePipeline:
    def __init__(self, algorithms: Iterable[NeuralInference]) -> None:
        self.algorithms = algorithms

    def append_simulations(self, theta: Tensor, x: Tensor, **kwargs):
        for algo in self.algorithms:
            algo.append_simulations(theta, x, **kwargs)

    def train(self, num_workers: Optional[int] = None):
        if num_workers is not None:
            if num_workers < 0:
                raise ValueError(
                    f"num_workers argument got negative value of {num_workers}"
                )

            pool = Pool(processes=num_workers)

            logger.info("Initializing training with %s workers", num_workers)
            _ = pool.map(_train, self.algorithms)
        else:
            logger.info("Initializing training sequentially")
            for algo in self.algorithms:
                _ = algo.train()

    def build_posterior(self, algorithm: Optional[NeuralInference] = None, **kwargs):
        if algorithm:
            algo = get_type_from_iterable(algorithm)
            return algo.build_posterior(**kwargs)
        else:
            for algo in self.algorithms:
                logger.info("Building posterior for %s", type(algo).__name__)
                posterior = algo.build_posterior(**kwargs)
                yield posterior
