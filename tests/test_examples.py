import pytest

import torch
from torch.distributions import Distribution

from sbisandbox.examples import (
    GaussianLinearToyModel,
    GaussianLinearUniformToyModel,
    TwoMoonsToyModel,
    SLCPToyModel,
    __models__,
)


def get_model(model: str):
    model_cls = (
        GaussianLinearToyModel,
        GaussianLinearUniformToyModel,
        TwoMoonsToyModel,
        SLCPToyModel,
    )
    return dict(zip(__models__, model_cls)).get(model)()


@pytest.mark.parametrize("model_name", __models__)
class TestToyModels:
    def test_prior(self, model_name):
        model = get_model(model_name)
        prior = model.prior
        assert isinstance(prior, Distribution)
        assert prior.event_shape == model.theta_event_shape

    @pytest.mark.parametrize("batch_size", (1, 10, 100))
    def test_simulator(self, model_name, batch_size):
        model = get_model(model_name)
        theta = torch.zeros((batch_size, model.params_dimensionality))
        samples = model.simulator(theta)
        assert samples.shape == (batch_size, *model.x_event_shape)
