import torch
from torch.distributions import MultivariateNormal

from .toymodel import ToyModel, MultivariateNormalMixin

class GaussianLinearToyModel(MultivariateNormalMixin, ToyModel):
    def __init__(self, nparams=10, ndata=10):
        super(ToyModel).__init__((nparams, ndata))
        self.nparams = 10
        self.ndata = 10
        self.prior_loc = torch.zeros(self.nparams)
        self.prior_covariance_matrix = 0.1 * torch.eye(self.nparams)

    @property
    def prior(self):
        return MultivariateNormal(self.prior_loc, covariance_matrix=self.prior_covariance_matrix)
    
    def loc(self, params: torch.Tensor) -> torch.Tensor:
        return params
    
    def covariance_matrix(self, params: torch.Tensor) -> torch.Tensor:
        return 0.1 * torch.eye(params.shape)