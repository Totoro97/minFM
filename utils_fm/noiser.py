from dataclasses import dataclass
import math
from typing import Protocol

import torch
import torch.nn as nn

from utils.config import BaseParams, ConfigurableModule
from utils.misc import DTYPE_MAP


class NoiserProtocol(Protocol):
    """Protocol defining the interface that a noiser module should implement."""

    def alpha_beta(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...


@dataclass
class FlowNoiserParams(BaseParams):
    """Parameters for FlowNoiser."""

    compute_dtype: str = "fp32"  # Internal computation dtype: "fp32", "fp16", "bf16"


class FlowNoiser(NoiserProtocol, nn.Module, ConfigurableModule[FlowNoiserParams]):
    def __init__(self, params: FlowNoiserParams) -> None:
        nn.Module.__init__(self)

        # Use the global DTYPE_MAP
        self.compute_dtype = DTYPE_MAP[params.compute_dtype]

    @classmethod
    def get_default_params(cls) -> FlowNoiserParams:
        """Return the default parameters for FlowNoiser."""
        return FlowNoiserParams()

    def alpha_beta(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute alpha and beta for the given timestep.
        t = 0 is clean data, t = 1 is pure noise.
        """
        alpha = 1 - t
        beta = t
        return alpha, beta

    def forward(
        self, x: torch.Tensor, x_datum_lens: torch.Tensor, t: torch.Tensor, rng: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply noise to the input tensor.
        """
        num_tokens = x.shape[0]
        t = torch.repeat_interleave(t, x_datum_lens, output_size=num_tokens)
        alpha, beta = self.alpha_beta(t)
        # Reshape for proper broadcasting: (N,) -> (N, 1)
        alpha = alpha.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        x_float = x.type(self.compute_dtype)
        gauss_noise = torch.randn(x_float.shape, device=x_float.device, dtype=x_float.dtype, generator=rng)
        x_noised = alpha * x_float + beta * gauss_noise
        v = gauss_noise - x_float
        return x_noised.type(x.dtype), v.type(x.dtype)


@dataclass
class TimeWarperParams(BaseParams):
    """Parameters for TimeWarper."""

    base_len: int = 256
    base_shift: float = 0.5
    max_len: int = 4096
    max_shift: float = 1.15


class TimeWarper(nn.Module, ConfigurableModule[TimeWarperParams]):
    def __init__(self, params: TimeWarperParams) -> None:
        nn.Module.__init__(self)
        self.base_len = params.base_len
        self.base_shift = params.base_shift
        self.max_len = params.max_len
        self.max_shift = params.max_shift

        # Precompute linear function coefficients
        self.slope = (self.max_shift - self.base_shift) / (self.max_len - self.base_len)
        self.intercept = self.base_shift - self.slope * self.base_len

    @classmethod
    def get_default_params(cls) -> TimeWarperParams:
        """Return the default parameters for TimeWarper."""
        return TimeWarperParams()

    def time_shift(self, mu: torch.Tensor, sigma: float, t: torch.Tensor) -> torch.Tensor:
        """Apply time shift transformation using exponential scaling."""
        exp_mu = torch.exp(mu)
        return exp_mu / (exp_mu + (1 / t - 1) ** sigma)

    def forward(self, t: torch.Tensor, target_len: int | torch.Tensor) -> torch.Tensor:
        """
        Apply time warping transformation based on target sequence length.

        Args:
            t (torch.Tensor): Time values tensor.
            target_len (int | torch.Tensor): Target sequence length. If int, applies the same length
                to all elements. If torch.Tensor, should have the same shape as t.

        Returns:
            torch.Tensor: Warped time values.
        """
        # Convert int to tensor with same shape as t
        if isinstance(target_len, int):
            target_len = torch.full_like(t, target_len, dtype=torch.int32)

        # Now target_len is always a tensor with the same shape as t
        mu = self.slope * target_len + self.intercept
        return self.time_shift(mu, 1.0, t)


def logit_normal_pdf(x: torch.Tensor, mu: float = 0.0, sigma: float = 1.0) -> torch.Tensor:
    eps = 1e-6
    x = x.clamp(min=eps, max=1 - eps)

    logit_x = torch.log(x / (1 - x))
    log_pdf = -torch.log(x * (1 - x)) - math.log(sigma * math.sqrt(2 * math.pi)) - 0.5 * ((logit_x - mu) / sigma) ** 2
    pdf = torch.exp(log_pdf)
    return pdf


@dataclass
class TimeWeighterParams(BaseParams):
    """Parameters for TimeWeighter."""

    use_logit_normal: bool = True
    mu: float = 0.0
    sigma: float = 1.0


class TimeWeighter(nn.Module, ConfigurableModule[TimeWeighterParams]):
    def __init__(self, params: TimeWeighterParams) -> None:
        nn.Module.__init__(self)
        self.use_logit_normal = params.use_logit_normal
        self.mu = params.mu
        self.sigma = params.sigma

    @classmethod
    def get_default_params(cls) -> TimeWeighterParams:
        """Return the default parameters for TimeWeighter."""
        return TimeWeighterParams()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute weights for time values.

        When use_logit_normal=True, computes weights using the logit normal probability density function
        with parameters mu and sigma. When use_logit_normal=False, returns uniform weights of 1.0.

        Args:
            t (torch.Tensor): Time values tensor of shape [batch_size,]. Values should be in range (0, 1).

        Returns:
            torch.Tensor: Weight values of shape [batch_size,]. When use_logit_normal=True, these are
            probability density values from the logit normal distribution. When use_logit_normal=False,
            these are uniform weights of 1.0.
        """
        if self.use_logit_normal:
            return logit_normal_pdf(t, mu=self.mu, sigma=self.sigma)
        else:
            return torch.ones_like(t)


def logit_normal_sample(
    size: tuple[int, ...],
    mu: float = 0.0,
    sigma: float = 1.0,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Sample from logit normal distribution.

    Args:
        size (tuple[int, ...]): Size of the output tensor as a tuple of integers.
        mu (float): Mean parameter of the underlying normal distribution.
        sigma (float): Standard deviation parameter of the underlying normal distribution.
        device (torch.device, optional): Device to place the tensor on.
        generator (torch.Generator, optional): Random number generator.

    Returns:
        torch.Tensor: Samples from logit normal distribution in range (0, 1).
    """
    normal_samples = torch.randn(size, device=device, dtype=torch.float32, generator=generator)
    logit_normal_samples = torch.sigmoid(normal_samples * sigma + mu)

    return logit_normal_samples


@dataclass
class TimeSamplerParams(BaseParams):
    """Parameters for TimeSampler."""

    use_logit_normal: bool = True
    mu: float = 0.0
    sigma: float = 1.0


class TimeSampler(nn.Module, ConfigurableModule[TimeSamplerParams]):
    def __init__(self, params: TimeSamplerParams) -> None:
        nn.Module.__init__(self)
        self.use_logit_normal = params.use_logit_normal
        self.mu = params.mu
        self.sigma = params.sigma

    @classmethod
    def get_default_params(cls) -> TimeSamplerParams:
        """Return the default parameters for TimeSampler."""
        return TimeSamplerParams()

    def forward(
        self,
        size: tuple[int, ...],
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        Sample time values from the specified distribution.

        When use_logit_normal=True, samples from the logit normal distribution with parameters
        mu and sigma. When use_logit_normal=False, samples uniformly from the range (0, 1).

        Args:
            size (tuple[int, ...]): Size of the output tensor as a tuple of integers.
            device (torch.device, optional): Device to place the tensor on.
            generator (torch.Generator, optional): Random number generator.

        Returns:
            torch.Tensor: Sampled time values of the specified size in range (0, 1).
        """
        if self.use_logit_normal:
            return logit_normal_sample(size, mu=self.mu, sigma=self.sigma, device=device, generator=generator)
        else:
            return torch.rand(size, device=device, dtype=torch.float32, generator=generator)
