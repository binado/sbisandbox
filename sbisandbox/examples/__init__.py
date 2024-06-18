# ruff: noqa: F401
from .gaussian_linear import GaussianLinearToyModel
from .gaussian_linear_uniform import GaussianLinearUniformToyModel
from .twomoons import TwoMoonsToyModel
from .slcp import SLCPToyModel

__models__ = ("Gaussian Linear", "Gaussian Linear Uniform", "Two moons", "SLCP")
