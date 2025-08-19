from dataclasses import dataclass, field
import itertools

import torch
import torch.nn as nn

from models.flux_utils import DoubleStreamBlock, LastLayer, MLPEmbedder, SingleStreamBlock, check_dtype
from models.transformer_utils import MultiAxisPositionalEncoding, rope_encoding, sinusoidal_encoding
from utils.comm import all_gather_with_padding
from utils.config import BaseParams, ConfigurableModule
from utils.pack import apply_interleave, plan_interleave
from knapformer import SequenceBalancer


@dataclass
class FluxDenoiserParams(BaseParams):
    d_model: int = 4096
    d_head: int = 128  # d_model // num_heads
    n_ds_blocks: int = 19
    n_ss_blocks: int = 38
    d_txt: int = 4096
    d_img: int = 64
    d_vec: int = 768
    rope_axis_dim: list[int] = field(default_factory=lambda: [16, 56, 56])  # tyx coordinates
    guidance_embed: bool = False


class FluxDenoiser(nn.Module, ConfigurableModule[FluxDenoiserParams]):
    def __init__(self, params: FluxDenoiserParams):
        nn.Module.__init__(self)
        self.params = params
        self.d_model = params.d_model
        self.d_head = params.d_head
        self.n_ds_blocks = params.n_ds_blocks
        self.n_ss_blocks = params.n_ss_blocks

        # d_txt -> d_model embedder
        self.txt_in = nn.Linear(params.d_txt, params.d_model)
        # d_img -> d_model embedder
        self.img_in = nn.Linear(params.d_img, params.d_model)
        # d_vec -> d_model embedder
        self.vector_in = MLPEmbedder(params.d_vec, params.d_model)
        if self.params.guidance_embed:
            self.guidance_in = MLPEmbedder(256, params.d_model)
        # diffusion timestep embedder
        self.t_posenc = lambda t: sinusoidal_encoding(t * 1000.0, dim=256)
        self.time_in = MLPEmbedder(256, params.d_model)

        # Just a placeholder, not used for pretraining
        # self.guidance_in = MLPEmbedder(in_dim=256, hidden_dim=params.d_model)
        # rope positional encoding
        self.rope_axis_dim = params.rope_axis_dim
        assert (
            sum(params.rope_axis_dim) == params.d_head
        ), f"sum(rope_axis_dim) must be equal to d_head, but got {sum(params.rope_axis_dim)} and {params.d_head}"
        self.rope_posenc = MultiAxisPositionalEncoding(
            rope_encoding, n_axes=len(params.rope_axis_dim), axis_dim=params.rope_axis_dim, flux_mode=True
        )

        # sequence processor
        self.double_blocks = nn.ModuleList()
        for _ in range(params.n_ds_blocks):
            self.double_blocks.append(DoubleStreamBlock(params.d_model, params.d_head, flux_mode=True))
        self.single_blocks = nn.ModuleList()
        for _ in range(params.n_ss_blocks):
            self.single_blocks.append(SingleStreamBlock(params.d_model, params.d_head, flux_mode=True))

        # d_model -> d_img output layer
        self.final_layer = LastLayer(params.d_model, params.d_img)

        self.init_weights()

        # parameters dtype during runtime
        self.runtime_parameter_dtype: torch.dtype | None = None

    def init_weights(self) -> None:
        nn.init.trunc_normal_(self.txt_in.weight, std=0.02)
        nn.init.constant_(self.txt_in.bias, 0.0)
        nn.init.trunc_normal_(self.img_in.weight, std=0.02)
        nn.init.constant_(self.img_in.bias, 0.0)
        self.vector_in.init_weights()
        self.time_in.init_weights()
        # self.guidance_in.init_weights()

        for block in itertools.chain(self.double_blocks, self.single_blocks):
            block.init_weights()

        self.final_layer.init_weights()

    @classmethod
    def get_default_params(cls) -> FluxDenoiserParams:
        """Return the default parameters for FluxDenoiser."""
        return FluxDenoiserParams()

    def _apply_blocks(
        self,
        txt_img: torch.Tensor,
        vec: torch.Tensor,
        position_ids: torch.Tensor,
        seq_lens: list[int] | torch.Tensor,
        seq_ids: torch.Tensor,
        modality_ids: torch.Tensor,
        txt_modality_id: int = 0,
        img_modality_id: int = 1,
        sequence_balancer: SequenceBalancer | None = None,
    ) -> torch.Tensor:
        """Apply all double and single stream blocks.

        The function encapsulates the repetitive tasks of:
        1. Computing RoPE positional encodings.
        2. Determining txt/img token indices from `modality_ids`.
        3. Sequentially applying every block in `double_blocks` then `single_blocks`.

        Args:
            txt_img: Sequence tensor.
            vec: Conditioning vector.
            position_ids: Corresponding position ID tensor (before RoPE).
            seq_lens: Sequence lengths for attention masking.
            seq_ids: Datum IDs mapping each token.
            modality_ids: Tensor indicating modality (txt/img) per token.
            txt_modality_id: Modality ID representing text tokens (default 0).
            img_modality_id: Modality ID representing image tokens (default 1).
            sequence_balancer: Optional balancer for distributed routing.
        """

        # 1. RoPE positional encoding
        rope_posenc = self.rope_posenc(position_ids)

        # 2. Compute modality-specific indices
        txt_indices, img_indices = (
            torch.nonzero(modality_ids == m_id, as_tuple=True)[0] for m_id in (txt_modality_id, img_modality_id)
        )

        # 3. Sequentially apply blocks
        for block in itertools.chain(self.double_blocks, self.single_blocks):
            txt_img = block(
                txt_img,
                vec,
                rope_posenc,
                seq_lens,
                seq_ids,
                txt_indices,
                img_indices,
                sequence_balancer,
            )

        return txt_img

    def forward(
        self,
        txt: torch.Tensor,
        txt_datum_lens: torch.Tensor,
        txt_position_ids: torch.Tensor,
        img: torch.Tensor,
        img_datum_lens: torch.Tensor,
        img_position_ids: torch.Tensor,
        t: torch.Tensor,
        vec: torch.Tensor,
        guidance: torch.Tensor | None = None,
        sequence_balancer: SequenceBalancer | None = None,
    ) -> torch.Tensor:
        """
        txt: (l1+l2+...+ln, d_txt)
        txt_datum_lens: (n,)
        txt_position_ids: (l1+l2+...+ln, d_position)
        img: (l1+l2+...+ln, d_img)
        img_datum_lens: (n,)
        img_position_ids: (l1+l2+...+ln, d_position)
        t: (n,); must not be in torch.bfloat16
        vec: (n, d_vec)
        guidance: (n,) | None
        """
        check_dtype(txt, txt_datum_lens, txt_position_ids, img, img_datum_lens, img_position_ids, t, vec, guidance)

        device, orig_img_dtype = img.device, img.dtype
        if self.runtime_parameter_dtype is None:
            self.runtime_parameter_dtype = next(self.parameters()).dtype

        # Convert to runtime parameter dtype
        txt, img, vec = (
            txt.type(self.runtime_parameter_dtype),
            img.type(self.runtime_parameter_dtype),
            vec.type(self.runtime_parameter_dtype),
        )  # This is safe as usually runtime_parameter_dtype is bfloat16

        if guidance is not None:
            guidance = guidance.type(self.runtime_parameter_dtype)

        # Step 1: merge global vec and time embedding and map d_txt, d_img to d_model
        t_enc = self.t_posenc(t).type(self.runtime_parameter_dtype)
        vec = self.vector_in(vec) + self.time_in(t_enc)
        if self.params.guidance_embed and guidance is not None:
            guidance = self.guidance_in(self.t_posenc(guidance).type(self.runtime_parameter_dtype))
            vec = vec + guidance

        txt, img = self.txt_in(txt), self.img_in(img)

        # Step 2: interleave txt and img sequences
        txt_modality_id, img_modality_id = 0, 1
        pack_meta = plan_interleave(
            [txt_datum_lens, img_datum_lens],
            [txt_modality_id, img_modality_id],
            device=device,
        )
        txt_img = apply_interleave([txt, img], pack_meta)
        txt_img_position_ids = apply_interleave([txt_position_ids, img_position_ids], pack_meta)

        if sequence_balancer is None:
            datum_lens, datum_ids, modality_ids = pack_meta.datum_lens, pack_meta.datum_ids, pack_meta.modality_ids
            txt_img = self._apply_blocks(
                txt_img,
                vec,
                txt_img_position_ids,
                datum_lens,
                datum_ids,
                modality_ids,
                txt_modality_id,
                img_modality_id,
                sequence_balancer,
            )
        else:
            # Step 3: sequence balancer plan routing
            sequence_balancer.plan_routing(pack_meta.datum_lens, txt_img.shape[-1])

            # Global datum IDs
            tmp_datum_ids = []
            for seq_len, seq_id in zip(
                sequence_balancer.gpu_id2seq_lens[sequence_balancer.this_gpu_id],
                sequence_balancer.gpu_id2seq_ids[sequence_balancer.this_gpu_id],
                strict=True,
            ):
                tmp_datum_ids.extend([seq_id] * seq_len)
            datum_ids = torch.tensor(tmp_datum_ids, device=device)

            # Global vec (b1+b2+...+bg, d_vec), where g is the number of GPUs
            global_vec = all_gather_with_padding(vec, group=sequence_balancer.balance_process_group)

            # Step 4: real routing of txt_img, and its metadata
            modality_ids, datum_ids = pack_meta.modality_ids.unsqueeze(-1), datum_ids.unsqueeze(-1)
            result = sequence_balancer.route(
                txt_img,
                [txt_img_position_ids, modality_ids, datum_ids],
                force_fp32_buffer=True,
            )
            bala_chunk_lens, bala_txt_img, (bala_txt_img_position_ids, bala_modality_ids, bala_datum_ids) = result  # type: ignore[misc]
            bala_modality_ids, bala_datum_ids = bala_modality_ids.squeeze(-1), bala_datum_ids.squeeze(-1)

            # Step 5: apply blocks (including RoPE + modality indices inside)
            bala_txt_img = self._apply_blocks(
                bala_txt_img,
                global_vec,
                bala_txt_img_position_ids,
                bala_chunk_lens,
                bala_datum_ids,
                bala_modality_ids,
                txt_modality_id,
                img_modality_id,
                sequence_balancer,
            )

            # Step 6: reverse route
            _, txt_img, _ = sequence_balancer.reverse_route(bala_txt_img)

        # Step 7: get the original img
        img_indices = pack_meta.modality_id2indices[img_modality_id]
        img, img_datum_ids = txt_img[img_indices], pack_meta.datum_ids[img_indices]
        img = self.final_layer(img, vec, img_datum_ids)
        return img.type(orig_img_dtype)
