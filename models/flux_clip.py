from dataclasses import dataclass
import os
from pathlib import Path

import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer

from utils.config import BaseParams, ConfigurableModule
from utils.log import get_logger
from utils.misc import DTYPE_MAP, Float32MatmulPrecision

logger = get_logger(__name__)


@dataclass
class CLIPEmbedderParams(BaseParams):
    version: str = "openai/clip-vit-large-patch14"
    max_length: int = 77
    dtype: str = "bf16"
    compile: bool = True
    float32_matmul_precision: str = "high"  # ["highest", "high", "medium"]


class CLIPEmbedder(nn.Module, ConfigurableModule[CLIPEmbedderParams]):
    def __init__(self, params: CLIPEmbedderParams) -> None:
        nn.Module.__init__(self)
        self.params = params
        self.max_length = params.max_length

        # Check for frozen cache directory environment variable
        minfm_cache_dir = os.environ.get("MINFM_CACHE_DIR")
        if minfm_cache_dir is not None:
            cache_path = Path(minfm_cache_dir) / params.version
            if not cache_path.exists():
                raise ValueError(f"Cache directory {cache_path} does not exist")

            self.tokenizer = CLIPTokenizer.from_pretrained(cache_path / "tokenizer", max_length=params.max_length)
            self.hf_module = CLIPTextModel.from_pretrained(cache_path / "model", torch_dtype=DTYPE_MAP[params.dtype])
        else:
            self.tokenizer = CLIPTokenizer.from_pretrained(params.version, max_length=params.max_length)
            self.hf_module = CLIPTextModel.from_pretrained(params.version, torch_dtype=DTYPE_MAP[params.dtype])

        self.hf_module = self.hf_module.eval().requires_grad_(False)

        self.float32_matmul_precision = params.float32_matmul_precision
        if params.compile:
            with Float32MatmulPrecision(self.float32_matmul_precision):
                self.hf_module = torch.compile(self.hf_module)

    @classmethod
    def get_default_params(cls) -> CLIPEmbedderParams:
        """Return the default parameters for CLIPEmbedder."""
        return CLIPEmbedderParams()

    def forward(self, text: list[str]) -> torch.Tensor:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = batch_encoding["input_ids"]  # (b, l)
        attention_mask = batch_encoding["attention_mask"]  # (b, l)

        device = self.hf_module.device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with Float32MatmulPrecision(self.float32_matmul_precision):
            outputs = self.hf_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
            )

        # CLIP uses pooler_output instead of last_hidden_state
        text_embedding: torch.Tensor = outputs["pooler_output"]  # (b, d)

        return text_embedding
