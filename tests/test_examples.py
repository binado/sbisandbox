import pytest
from itertools import product

from torch.distributions import Distribution

from sbisandbox.benchmarks import (
    GaussianLinearBenchmark,
    GaussianLinearUniformBenchmark,
    GaussianMixtureBenchmark,
    TwoMoonsBenchmark,
    SLCPBenchmark,
    __models__,
)


def get_model(model: str):
    model_cls = (
        GaussianLinearBenchmark,
        GaussianLinearUniformBenchmark,
        GaussianMixtureBenchmark,
        TwoMoonsBenchmark,
        SLCPBenchmark,
    )
    return dict(zip(__models__, model_cls)).get(model)()


@pytest.mark.parametrize("model_name,batch_size", product(__models__, (1, 10, 100)))
class TestBenchmarks:
    def test_prior(self, model_name, batch_size):
        model = get_model(model_name)
        prior = model.prior
        assert isinstance(prior, Distribution)
        assert prior.event_shape == model.theta_event_shape
        prior_samples = prior.sample((batch_size,))
        assert prior_samples.shape == (batch_size, *model.theta_event_shape)

    def test_simulator(self, model_name, batch_size):
        model = get_model(model_name)
        theta = model.prior.sample((batch_size,))
        samples = model.simulator(theta)
        assert samples.shape == (batch_size, *model.x_event_shape)
