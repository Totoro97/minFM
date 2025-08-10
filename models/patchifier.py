from dataclasses import dataclass

from einops import rearrange
import torch

from utils.config import BaseParams, ConfigurableModule


@dataclass
class PatchifierParams(BaseParams):
    patch_size: tuple[int, int, int] = (1, 2, 2)  # [frames, height, width] - DiT typical
    vae_latent_channels: int = 16  # VAE latent channels
    vae_compression_factors: tuple[int, int, int] = (1, 8, 8)  # VAE compression factors [frames, height, width]


class Patchifier(ConfigurableModule[PatchifierParams]):
    def __init__(self, params: PatchifierParams) -> None:
        self.params = params
        self.patch_size = params.patch_size
        self.vae_latent_channels = params.vae_latent_channels
        self.vae_compression_factors = params.vae_compression_factors

        self.pf, self.ph, self.pw = self.patch_size
        self.vec_dim = self.pf * self.ph * self.pw * self.vae_latent_channels

        self.b, self.c, self.f, self.h, self.w = None, None, None, None, None
        self.gf, self.gh, self.gw = None, None, None
        self.device = None

    @classmethod
    def get_default_params(cls) -> PatchifierParams:
        """Return the default parameters for Patchifier."""
        return PatchifierParams()

    def patchify(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (b, c, f, h, w)
        """
        assert (
            x.shape[1] == self.vae_latent_channels
        ), f"vae_latent_channels must match, but got {x.shape[1]} and {self.vae_latent_channels}"

        b, c, f, h, w = x.shape
        self.b, self.f, self.c, self.h, self.w = b, f, c, h, w  # type: ignore
        self.gf, self.gh, self.gw = f // self.pf, h // self.ph, w // self.pw  # type: ignore
        self.device = x.device  # type: ignore

        x = rearrange(
            x,
            "b c (gf pf) (gh ph) (gw pw) -> (b gf gh gw) (c pf ph pw)",
            b=self.b,
            gf=self.gf,
            pf=self.pf,
            gh=self.gh,
            ph=self.ph,
            gw=self.gw,
            pw=self.pw,
        )
        datum_lens = torch.full((self.b,), self.gf * self.gh * self.gw, device=self.device, dtype=torch.int32)  # type: ignore
        position_ids = self.get_position_ids()
        return x, datum_lens, position_ids

    def get_num_position_axes(self) -> int:
        """
        Get the number of position axes for the patchified image.
        """
        return len(self.patch_size)

    def get_position_ids(self, is_img: bool = True) -> torch.Tensor:
        """
        Get position ids for the patchified image.
        """
        # Create coordinates directly without meshgrid to minimize kernel launches
        total_elements_per_datum = self.gf * self.gh * self.gw  # type: ignore
        total_elements = self.b * total_elements_per_datum

        # Create flat indices for all positions
        indices = torch.arange(total_elements, device=self.device, dtype=torch.int32)

        # Convert flat indices to 3D coordinates using integer arithmetic
        # Align with Flux's convention (t, y, x order)
        datum_idx = indices % total_elements_per_datum

        # Extract t, y, x coordinates from spatial index
        remaining = datum_idx % (self.gh * self.gw)  # type: ignore
        y_coords = remaining // self.gw
        x_coords = remaining % self.gw

        # Handle time coordinate efficiently based on is_img
        if is_img:
            t_coords = torch.zeros_like(datum_idx)
        else:
            t_coords = datum_idx // (self.gh * self.gw)  # type: ignore

        # Stack coordinates into final tensor
        tyx = torch.stack([t_coords, y_coords, x_coords], dim=-1)
        return tyx

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (b gf gh gw) (pf ph pw c)
        """
        x = rearrange(
            x,
            "(b gf gh gw) (c pf ph pw) -> b c (gf pf) (gh ph) (gw pw)",
            b=self.b,
            gf=self.gf,
            gh=self.gh,
            gw=self.gw,
            pf=self.pf,
            ph=self.ph,
            pw=self.pw,
        )

        # reset
        self.b, self.c, self.f, self.h, self.w = None, None, None, None, None
        self.gf, self.gh, self.gw = None, None, None
        self.device = None

        return x
