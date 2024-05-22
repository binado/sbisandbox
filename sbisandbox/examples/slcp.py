import torch

from .toymodel import ToyModel, UniformPriorMixin, MultivariateNormalMixin

class SLCPToyModel(UniformPriorMixin, MultivariateNormalMixin, ToyModel):
    def __init__(self) -> None:
        super(ToyModel).__init__((5, 8))

    @property
    def low(self) -> torch.Tensor:
        return -1 * torch.ones(self.params_dimensionality)
    
    @property
    def high(self) -> torch.Tensor:
        return torch.ones(self.params_dimensionality)
    
    def loc(self, params: torch.Tensor) -> torch.Tensor:
        return params[:2]
    
    def covariance_matrix(self, params: torch.Tensor) -> torch.Tensor:
        s1  = params[2].item() ** 2
        s2 =  params[3].item() ** 2
        rho = torch.atan(params[-1]).item()
        return torch.tensor([
            [s1 ** 2, rho * s1 * s2],
            [rho * s1 * s2, s2 ** 2]
        ])
