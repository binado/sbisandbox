# ruff: noqa: F401
from .gaussian_linear import GaussianLinearToyModel
from .gaussian_linear_uniform import GaussianLinearUniformToyModel
from .gaussian_mixture import GaussianMixtureToyModel
from .twomoons import TwoMoonsToyModel
from .slcp import SLCPToyModel

__models__ = (
    "Gaussian Linear",
    "Gaussian Linear Uniform",
    "Gaussian Mixture" "Two Moons",
    "SLCP",
)
