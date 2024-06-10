from sbi.inference import MCMCPosterior

from .examples.toymodel import ToyModel


def get_sampler(toy_model: ToyModel, method="nuts_pyro", **kwargs):
    return MCMCPosterior(
        potential_fn=toy_model.log_prob,
        proposal=toy_model.prior,
        method=method,
        **kwargs,
    )
