from dataclasses import dataclass
import os
from pathlib import Path

import torch
import torch.nn as nn
from transformers import T5EncoderModel, T5Tokenizer

from utils.config import BaseParams, ConfigurableModule
from utils.log import get_logger
from utils.misc import DTYPE_MAP, Float32MatmulPrecision

logger = get_logger(__name__)


@dataclass
class T5EmbedderParams(BaseParams):
    version: str = "google/t5-v1_1-xxl"
    max_length: int = 256
    dtype: str = "bf16"
    compile: bool = True
    float32_matmul_precision: str = "high"  # ["highest", "high", "medium"]
    padding_side: str = "right"  # ["left", "right"]
    attn_mask_padding: bool = True  # Whether to mask padding tokens in attention
    output_exclude_padding: bool = True  # Whether to exclude padding tokens from output


class T5Embedder(nn.Module, ConfigurableModule[T5EmbedderParams]):
    def __init__(self, params: T5EmbedderParams) -> None:
        nn.Module.__init__(self)
        self.params = params
        self.max_length = params.max_length

        # Check for frozen cache directory environment variable
        minfm_cache_dir = os.environ.get("MINFM_CACHE_DIR")
        if minfm_cache_dir is not None:
            cache_path = Path(minfm_cache_dir) / params.version
            if not cache_path.exists():
                raise ValueError(f"Cache directory {cache_path} does not exist")

            self.tokenizer = T5Tokenizer.from_pretrained(cache_path / "tokenizer", max_length=params.max_length)
            self.hf_module = T5EncoderModel.from_pretrained(cache_path / "model", torch_dtype=DTYPE_MAP[params.dtype])
        else:
            self.tokenizer = T5Tokenizer.from_pretrained(params.version, max_length=params.max_length)
            self.hf_module = T5EncoderModel.from_pretrained(params.version, torch_dtype=DTYPE_MAP[params.dtype])

        # Set the tokenizer padding side based on configuration
        self.tokenizer.padding_side = params.padding_side
        self.hf_module = self.hf_module.eval().requires_grad_(False)

        self.float32_matmul_precision = params.float32_matmul_precision
        if params.compile:
            with Float32MatmulPrecision(self.float32_matmul_precision):
                self.hf_module = torch.compile(self.hf_module)

    @classmethod
    def get_default_params(cls) -> T5EmbedderParams:
        """Return the default parameters for T5Embedder."""
        return T5EmbedderParams()

    def forward(self, text: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        # Padding token id is 0
        # An empty string is mapped to end-of-sequence (EOS) token whose id is 1
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

        # Store the original attention mask for output filtering
        original_attention_mask = attention_mask.clone()

        # If attn_mask_padding is False, use all-ones attention mask for the model
        if not self.params.attn_mask_padding:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

        with Float32MatmulPrecision(self.float32_matmul_precision):
            outputs = self.hf_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
            )  # (b, l, d)

        text_embedding = outputs["last_hidden_state"].flatten(0, 1)  # (b*l, d)

        # Determine which mask to use for output filtering
        if self.params.output_exclude_padding:
            output_mask = original_attention_mask
        else:
            output_mask = torch.ones_like(original_attention_mask)

        text_embedding_mask = output_mask.bool()  # (b, l)
        text_datum_lens = output_mask.sum(dim=-1)  # (b,)
        text_embedding_mask = text_embedding_mask.flatten(0, 1)  # (b*l,)
        text_embedding = text_embedding[text_embedding_mask]  # (l1+l2+...+ln, d)

        return text_embedding, text_datum_lens
