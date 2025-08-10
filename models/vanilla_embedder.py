from dataclasses import dataclass, field

import torch
import torch.nn as nn

from utils.config import BaseParams, ConfigurableModule
from utils.log import get_logger
from utils.misc import load_pt_data_from_path

logger = get_logger(__name__)


@dataclass
class VanillaEmbedderParams(BaseParams):
    vocab_size: int = 1001  # Size of vocabulary, for imagenet it's 1000 + 1 for empty string
    embedding_dim: int = 768  # Output embedding dimension
    return_datum_lens: bool = False
    embeddings_path: str = "$MINFM_DATA_DIR/imagenet/ilsvrc2012_meta.pt::clip_embeddings"
    txt_to_label_path: str = "$MINFM_DATA_DIR/imagenet/ilsvrc2012_meta.pt::txt_to_label"
    special_prompts: list[str] = field(default_factory=lambda: ["", "DUMMY_PROMPT"])


class VanillaEmbedder(nn.Module, ConfigurableModule[VanillaEmbedderParams]):
    def __init__(self, params: VanillaEmbedderParams) -> None:
        nn.Module.__init__(self)
        self.params = params

        # Create the embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=params.vocab_size,
            embedding_dim=params.embedding_dim,
        )

        self.txt_to_label = None
        embeddings = load_pt_data_from_path(self.params.embeddings_path)
        txt_to_label = load_pt_data_from_path(self.params.txt_to_label_path)

        assert (
            embeddings.shape == self.embedding.weight.shape
        ), f"embeddings shape mismatch: {embeddings.shape} != {self.embedding.weight.shape}"

        self.embedding.weight.data.copy_(embeddings)
        self.txt_to_label = txt_to_label

    @classmethod
    def get_default_params(cls) -> VanillaEmbedderParams:
        """Return the default parameters for VanillaEmbedder."""
        return VanillaEmbedderParams()

    def forward(self, text: list[str]) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the vanilla embedder.

        Args:
            text: List of input strings

        Returns:
            torch.Tensor: Embeddings of shape (batch_size, embedding_dim)
        """

        # Do some sanity checks
        for txt in text:
            assert txt in self.params.special_prompts or txt in self.txt_to_label, f"Invalid text: {txt}"

        # If text is in txt_to_label, use the index, otherwise use the index of empty string
        indices = [self.txt_to_label[t] if t in self.txt_to_label else self.txt_to_label[""] for t in text]

        # Convert to tensor
        input_ids = torch.tensor(indices, dtype=torch.long, device=self.embedding.weight.device)

        # Get embeddings
        embeddings = self.embedding(input_ids)

        if not self.params.return_datum_lens:
            return embeddings  # type: ignore[no-any-return]
        else:
            return embeddings, torch.ones(len(text), dtype=torch.long, device=self.embedding.weight.device)

        raise ValueError(f"Invalid output style: {self.params.output_style}")
