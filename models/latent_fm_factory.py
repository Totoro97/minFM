"""
Model factory for creating configurable LatentFM components.
"""

from copy import deepcopy
from typing import Any

import torch

from models.latent_fm import LatentFM
from utils.config import create_component
from utils.log import get_logger
from knapformer import SequenceBalancer

logger = get_logger(__name__)


def create_latent_fm(config: dict[str, Any], device: torch.device, create_ema: bool = True) -> LatentFM:
    """Create LatentFM with all components based on config."""
    logger.info("Creating LatentFM components...")

    ########## VAE ##########
    logger.info("Creating VAE...")
    vae_config = config["model"]["vae"]
    fsdp_spec = vae_config.get("fsdp", None)
    vae = create_component(vae_config["module"], vae_config["params"], fsdp_spec)
    vae = vae.to(device=device, dtype=torch.bfloat16)
    for param in vae.parameters():
        param.requires_grad = False

    ########## Text Encoder ##########
    logger.info("Creating Text Encoder...")
    text_encoder_config = config["model"]["text_encoder"]
    fsdp_spec = text_encoder_config.get("fsdp", None)
    text_encoder = create_component(text_encoder_config["module"], text_encoder_config["params"], fsdp_spec)
    text_encoder = text_encoder.to(device, dtype=torch.bfloat16)  # noops
    for param in text_encoder.parameters():
        param.requires_grad = False

    ########## Clip Encoder ##########
    clip_encoder = None
    if "clip_encoder" in config["model"]:
        logger.info("Creating Clip Encoder...")
        clip_encoder_config = config["model"]["clip_encoder"]
        fsdp_spec = clip_encoder_config.get("fsdp", None)
        clip_encoder = create_component(clip_encoder_config["module"], clip_encoder_config["params"], fsdp_spec)
        clip_encoder = clip_encoder.to(device, dtype=torch.bfloat16)  # noops
        for param in clip_encoder.parameters():
            param.requires_grad = False

    ########## Patchifier ##########
    logger.info("Creating Patchifier...")
    patchifier_config = config["model"]["patchifier"]
    patchifier = create_component(patchifier_config["module"], patchifier_config["params"])

    ########## Denoiser ##########
    logger.info("Creating Denoiser...")
    denoiser_config = config["model"]["denoiser"]
    fsdp_spec = denoiser_config.get("fsdp", None)
    denoiser = create_component(denoiser_config["module"], denoiser_config["params"], fsdp_spec)
    denoiser = denoiser.to(device)  # noops
    denoiser.init_weights()
    for param in denoiser.parameters():
        param.requires_grad = True

    ########## EMA Denoiser ##########
    ema_denoiser = None
    if create_ema:
        logger.info("Creating EMA Denoiser...")
        ema_denoiser_config = config["model"]["denoiser"]
        fsdp_spec = ema_denoiser_config.get("fsdp", None)
        if fsdp_spec is not None:
            # we need to always reshard after forward for ema_denoiser, as it's never back-propagated through
            fsdp_spec = deepcopy(fsdp_spec)
            fsdp_spec["reshard_after_forward_policy"] = "always"

        ema_denoiser = create_component(ema_denoiser_config["module"], ema_denoiser_config["params"], fsdp_spec)
        ema_denoiser = ema_denoiser.to(device)  # noops
        # No need to init weights for ema_denoiser, as it will be copied from denoiser at training start
        for param in ema_denoiser.parameters():
            param.requires_grad = False  # we will directly modify ema_denoiser parameters in training

    ########## Time Sampler ##########
    logger.info("Creating Time Sampler...")
    time_sampler_config = config["model"]["time_sampler"]
    time_sampler = create_component(time_sampler_config["module"], time_sampler_config["params"])

    ########## Time Warper ##########
    logger.info("Creating Time Warper...")
    time_warper_config = config["model"]["time_warper"]
    time_warper = create_component(time_warper_config["module"], time_warper_config["params"])

    ########## Time Weighter ##########
    logger.info("Creating Time Weighter...")
    time_weighter_config = config["model"]["time_weighter"]
    time_weighter = create_component(time_weighter_config["module"], time_weighter_config["params"])

    ########## Flow Noiser ##########
    logger.info("Creating Flow Noiser...")
    flow_noiser_config = config["model"]["flow_noiser"]
    flow_noiser = create_component(flow_noiser_config["module"], flow_noiser_config["params"])

    ########## DIT Balancer ##########
    balancer_config = config["model"]["balancer"]
    dit_balancer = None
    if balancer_config["use_dit_balancer"]:
        logger.info("Creating DIT Balancer...")
        dit_balancer = SequenceBalancer(balancer_config["dit_balancer_specs"], balancer_config["dit_balancer_gamma"])

    ########## LatentFM ##########
    latent_fm = LatentFM(
        text_encoder=text_encoder,
        clip_encoder=clip_encoder,
        vae=vae,
        patchifier=patchifier,
        denoiser=denoiser,
        ema_denoiser=ema_denoiser,
        time_sampler=time_sampler,
        time_warper=time_warper,
        time_weighter=time_weighter,
        flow_noiser=flow_noiser,
        dit_balancer=dit_balancer,
    )
    latent_fm.summarize()
    return latent_fm
