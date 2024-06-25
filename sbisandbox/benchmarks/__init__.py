# ruff: noqa: F401
from .gaussian_linear import GaussianLinearBenchmark
from .gaussian_linear_uniform import GaussianLinearUniformBenchmark
from .gaussian_mixture import GaussianMixtureBenchmark
from .twomoons import TwoMoonsBenchmark
from .slcp import SLCPBenchmark

__models__ = (
    "Gaussian Linear",
    "Gaussian Linear Uniform",
    "Gaussian Mixture" "Two Moons",
    "SLCP",
)
