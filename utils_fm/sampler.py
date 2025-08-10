from collections.abc import Callable
from typing import Literal

import torch
import torch.nn as nn

from utils.pack import pack_reduce
from utils_fm.noiser import NoiserProtocol


def energy_preserve_cfg(
    pos_img_v: torch.Tensor, neg_img_v: torch.Tensor, img_v_datum_lens: torch.Tensor, cfg_scale: float
) -> torch.Tensor:
    """Apply classifier-free guidance while preserving energy of the velocity field.

    Based on:
        Zhang et al. 2024. EP-CFG: Energy-Preserving Classifier-Free Guidance.
        https://arxiv.org/abs/2412.09966

    This function combines positive and negative velocity predictions using classifier-free
    guidance, then rescales the result to preserve the energy (L2 norm) of the positive
    prediction. The energy preservation is crucial to avoid over-saturation effects that
    can occur with standard CFG, helping maintain stable generation quality.

    Args:
        pos_img_v: Positive (conditioned) velocity predictions for images.
        neg_img_v: Negative (unconditioned) velocity predictions for images.
        img_v_datum_lens: Length of each sequence in the batch for proper energy calculation.
        cfg_scale: Classifier-free guidance scale factor. Higher values increase conditioning strength.

    Returns:
        Energy-preserved classifier-free guidance velocity field that avoids over-saturation.
    """
    pos_img_v_energy = pack_reduce(pos_img_v**2, img_v_datum_lens, reduction="sum")
    cfg_img_v = pos_img_v + (cfg_scale - 1.0) * (pos_img_v - neg_img_v)
    cfg_img_v_energy = pack_reduce(cfg_img_v**2, img_v_datum_lens, reduction="sum")
    scale = (pos_img_v_energy / cfg_img_v_energy) ** 0.5
    num_tokens = pos_img_v.shape[0]
    scale = torch.repeat_interleave(scale, img_v_datum_lens, output_size=num_tokens)
    return cfg_img_v * scale[:, None]


class FlowSampler:
    def __init__(
        self,
        velocity_model: Callable,
        noiser: NoiserProtocol,
        t_warper: nn.Module,
        sample_method: Literal["euler", "ddim"] = "ddim",
    ) -> None:
        super().__init__()
        self.velocity_model = velocity_model
        self.noiser = noiser
        self.t_warper = t_warper
        self.sample_method = sample_method

    def __call__(
        self,
        x: torch.Tensor,
        x_datum_lens: torch.Tensor,
        x_position_ids: torch.Tensor,
        num_steps: int,
        warp_len: int,
        rng: torch.Generator | None = None,
        eta: float = 0,
    ) -> torch.Tensor:
        assert x.dtype == torch.float32, f"x must be float32, got {x.dtype}"
        timesteps = torch.linspace(1, 0, num_steps + 1, device=x.device, dtype=torch.float32)

        n = x_datum_lens.shape[0]
        for t, next_t in zip(timesteps[:-1], timesteps[1:], strict=True):
            t_warped = self.t_warper(t, warp_len)
            next_t_warped = self.t_warper(next_t, warp_len)

            v = self.velocity_model(x, x_datum_lens, x_position_ids, t_warped.repeat(n))  # eps - x0

            if self.sample_method == "euler":
                x = x + v * (next_t_warped - t_warped)
            else:
                alpha, beta = self.noiser.alpha_beta(t_warped)
                next_alpha, next_beta = self.noiser.alpha_beta(next_t_warped)

                x_0, eps = x - beta * v, x + alpha * v

                # Note: Below is a huristic predictor-corrector SDE solver step.
                sde_beta = (next_beta - eta * (beta - next_beta) * (next_beta / beta)).clamp(min=0.0)
                gauss_noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=rng)
                x = next_alpha * x_0 + sde_beta * eps + (next_beta**2 - sde_beta**2).sqrt() * gauss_noise

        return x
