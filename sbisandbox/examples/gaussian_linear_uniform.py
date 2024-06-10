import torch

from ..toymodel import ToyModel, UniformPriorMixin, MultivariateNormalMixin


class GaussianLinearUniformToyModel(
    UniformPriorMixin, MultivariateNormalMixin, ToyModel
):
    def __init__(self, nparams=10, ndata=10):
        super(ToyModel).__init__((nparams, ndata))

    @property
    def low(self):
        return -1 * torch.ones(self.params_dimensionality)

    @property
    def high(self):
        return torch.ones(self.params_dimensionality)

    def loc(self, params: torch.Tensor) -> torch.Tensor:
        return params

    def covariance_matrix(self, params: torch.Tensor) -> torch.Tensor:
        return 0.1 * torch.eye(params.shape)
