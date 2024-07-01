from typing import Tuple
import math

import pyro.distributions as dist
import torch
from torch import nn, Tensor
from torch.distributions import Normal
from torchdiffeq import odeint_adjoint as odeint

from ..types import Shape
from ..benchmark import Benchmark, UniformPriorMixin


class LotkaVolterraODE(nn.Module):
    def __init__(self, theta: Tensor, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert theta.shape[-1] == 4
        self.theta = nn.Parameter(theta)

    def forward(self, t, state):
        prey, predator = state
        dprey = self.theta[..., 0] * prey - self.theta[..., 1] * prey * predator
        dpredator = (
            -self.theta[..., 2] * predator + self.theta[..., 3] * prey * predator
        )
        return dprey, dpredator


class LotkaVolterraBenchmark(UniformPriorMixin, Benchmark):
    r"""Lotka Volterra benchmark task.

    The parameters $\boldsymbol{\theta} = \{\alpha, \beta, \gamma, \delta\} \in \mathbb{R}^4$
    are sampled from

    $$ \alpha \sim \text{Lognormal}([-0.125, 0.5])$$

    $$ \beta \sim \text{Lognormal}([-3, 0.5])$$

    $$ \gamma \sim \text{Lognormal}([-0.125, 0.5])$$

    $$ \delta \sim \text{Lognormal}([-3, 0.5])$$

    The Lotka-Volterra equations characterize the evolution of two biological species:
    preys and predators, whose populations at a time $t$ are denoted by $x(t)$ and $y(t)$
    respectively. The two populations evolve as

    $$
    \frac{dx}{dt} = \alpha x - \beta xy
    $$

    $$
    \frac{dy}{dt} = -\gamma y + \delta xy
    $$

    We define our data as the population levels of each species at 10 equally spaced time intervals,
    $\boldsymbol{x} = \{(x(t_i), y(t_i))\}$ for $t_i \in \{t_f / 10, 2t_f / 10, \ldots, t_f\}$,
    where the equations are integrated from $t=0$ until $t=t_f$.

    <figure markdown="span">
      ![prey-predator evolution](../img/lk.png){ width="500" }
      <figcaption>Prey-predator populations for x(0) = 30 and y(0) = 1</figcaption>
    </figure>

    <figure markdown="span">
        ![prey-predator phase-space](../img/lk_phase_plot.png){ width="500" }
        <figcaption>Phase space plot for the same parameters and initial conditions</figcaption>
    </figure>

    The model has interesting dynamics, displaying two different fixed points on its phase space.
    The [wikipedia page](https://en.m.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations)
    is worth a read for a quick idea.
    """

    def __init__(self, y0=(30, 1), tmax: float = 50.0, npoints: int = 1000):
        super().__init__(theta_event_shape=(4,), x_event_shape=(10, 2))
        self.r_dist = Normal(0.1, 0.01)
        self.offset = torch.tensor([0.25, 0.0])
        self._y0 = y0
        self.npoints = npoints
        self.t = torch.linspace(0, tmax, npoints)

    def get_y0(self, theta: Tensor) -> Tuple[Tensor, Tensor]:
        return (
            torch.tensor(self._y0[0], dtype=torch.float32).expand((theta.shape[0],)),
            torch.tensor(self._y0[1], dtype=torch.float32).expand((theta.shape[0],)),
        )

    @property
    def x0(self):
        return torch.tensor([0, 0])

    @property
    def low(self):
        return -1 * torch.ones(self.theta_event_shape[0])

    @property
    def high(self):
        return torch.ones(self.theta_event_shape[0])

    def _pyro_model(self) -> Tensor:
        raise NotImplementedError

    def simulator(self, theta: torch.Tensor) -> torch.Tensor:
        model = LotkaVolterraODE(theta)
        y0 = self.get_y0(theta)
        prey, predator = odeint(model, y0, self.t)  # type: ignore
        n = self.npoints // 10
        return torch.stack((prey.t()[..., ::n], predator.t()[..., ::n]), dim=-1)  # type: ignore

    def get_posterior_samples(self, shape: Shape, x: torch.Tensor) -> torch.Tensor:
        x_to_shape = x.expand(shape + self.theta_event_shape)
        a = (
            dist.Uniform(low=-0.5 * math.pi, high=0.5 * math.pi)
            .sample(shape)
            .unsqueeze(-1)
        )
        r = dist.Normal(loc=0.1, scale=0.01).sample(shape).unsqueeze(-1)
        p = r * torch.cat((torch.cos(a), torch.sin(a)), dim=-1) + self.offset
        # Theta is rotated
        rot_t = p - x_to_shape

        # To perform the inverse rotation, we must first decide on the sign of \theta_1 + \theta_2
        u = torch.rand(shape)
        plus_rotated = torch.stack(
            (rot_t[..., 0] + rot_t[..., 1], rot_t[..., 0] - rot_t[..., 1]), dim=-1
        ) / math.sqrt(2)
        minus_rotated = torch.stack(
            (-rot_t[..., 0] + rot_t[..., 1], -rot_t[..., 0] - rot_t[..., 1]), dim=-1
        ) / math.sqrt(2)
        return torch.where(u.unsqueeze(-1) > 0.5, plus_rotated, minus_rotated)

    def loglike(self, params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        abssum = torch.abs(torch.sum(params, 1))
        diff = params[:, 0] - params[:, 1]
        # (-|p1 + p2|, p1 - p2) / sqrt(2)
        rotated_params = torch.stack((abssum, diff), dim=1) / math.sqrt(2)
        p = x + rotated_params - self.offset
        # r is normally distributed as N(0.1, 0.01)
        r = torch.sum(p**2)
        return self.r_dist.log_prob(r)
