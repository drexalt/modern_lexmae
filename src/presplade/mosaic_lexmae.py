import torch.nn as nn
from transformers import (
    AutoModelForMaskedLM,
    PreTrainedModel,
)
from typing import Any, Dict, List
import importlib
import torch
import copy
import math

from .lexmae_base import LexMAEBase


class _MosaicLayerWrapper(nn.Module):
    def __init__(self, core, config):
        super().__init__()
        self.core = core
        self.config = config
        self._current_alibi_size = int(config.alibi_starting_size)
        self.alibi = torch.zeros(
            (
                1,
                self.config.num_attention_heads,
                self._current_alibi_size,
                self._current_alibi_size,
            )
        )
        self.build_alibi(
            self.config.num_attention_heads, self._current_alibi_size, self.alibi.device
        )

    def build_alibi(self, n_heads, size, device):
        n_heads = self.config.num_attention_heads

        def _get_alibi_head_slopes(n_heads: int) -> List[float]:
            def get_slopes_power_of_2(n_heads: int) -> List[float]:
                start = 2 ** (-(2 ** -(math.log2(n_heads) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n_heads)]

            # In the paper, they only train models that have 2^a heads for some a. This function
            # has some good properties that only occur when the input is a power of 2. To
            # maintain that even when the number of heads is not a power of 2, we use a
            # workaround.
            if math.log2(n_heads).is_integer():
                return get_slopes_power_of_2(n_heads)

            closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
            slopes_a = get_slopes_power_of_2(closest_power_of_2)
            slopes_b = _get_alibi_head_slopes(2 * closest_power_of_2)
            slopes_b = slopes_b[0::2][: n_heads - closest_power_of_2]
            return slopes_a + slopes_b

        context_position = torch.arange(size, device=device)[:, None]
        memory_position = torch.arange(size, device=device)[None, :]
        relative_position = torch.abs(memory_position - context_position)
        # [n_heads, max_token_length, max_token_length]
        relative_position = relative_position.unsqueeze(0).expand(n_heads, -1, -1)
        slopes = torch.Tensor(_get_alibi_head_slopes(n_heads)).to(device)
        alibi = slopes.unsqueeze(1).unsqueeze(1) * -relative_position
        # [1, n_heads, max_token_length, max_token_length]
        alibi = alibi.unsqueeze(0)
        assert alibi.shape == torch.Size([1, n_heads, size, size])

        self._current_alibi_size = size
        self.alibi = alibi

    def forward(self, hidden, attention_mask=None, output_attentions=False):
        # ----- book-keeping for Mosaic-style unpadding --------------------
        bs, L, _ = hidden.shape
        device = hidden.device
        if attention_mask is None:
            attention_mask = hidden.new_ones(bs, L, dtype=torch.long)

        lengths = attention_mask.sum(1).to(torch.int32)  # [bs]
        cu_seqlens = torch.cat(
            [torch.zeros(1, device=device, dtype=torch.int32), lengths.cumsum(0)],
            0,  # [bs+1]
        )
        max_len = int(lengths.max())
        attn_mask_slice = attention_mask[:, :max_len]  # [bs, max_len]
        flat_idx = torch.nonzero(attn_mask_slice.flatten(), as_tuple=False).squeeze()

        self.build_alibi(self.config.num_attention_heads, max_len, device)
        bias_alibi = self.alibi.expand(bs, -1, -1, -1)

        # -----------------------------------------------------------------

        # 1) UNPAD --------------------------  [total_nnz, h]
        unpadded = hidden.view(-1, self.config.hidden_size)[flat_idx]

        # extended_attention_mask = (1.0 - attn_mask_slice[:, None, None, :]) * -10000.0
        bias = (-10000.0 * (1 - attn_mask_slice[:, None, None, :])) + bias_alibi

        # 2) run Mosaic layer (expects un-padded inputs)
        out = self.core(
            unpadded,  # hidden_states
            cu_seqlens,  # cu_seqlens
            max_len,  # seqlen
            None,  # subset_idx
            flat_idx,  # indices
            attn_mask_slice,  # attn_mask
            bias,  # bias
        )

        # 3) PAD BACK so LexMAEBase can keep using the HF convention
        padded = hidden.clone()
        padded.view(-1, self.config.hidden_size)[flat_idx] = out

        return (padded, None)  # mimic HF (hidden_states, attn)


class MosaicLexMAE(LexMAEBase):
    """
    The only thing we have to teach LexMAEBase about BERT is how to
    build **one** transformer block for the decoder tower.
    """

    def _build_decoder_layer(self) -> nn.Module:
        """Return a single `BertLayer` initialised with the encoder config."""

        BertLayer = getattr(
            importlib.import_module(self.encoder.__class__.__module__), "BertLayer"
        )(self.encoder.config)

        return _MosaicLayerWrapper(BertLayer, self.encoder.config)

    def _build_decoder_head(self) -> nn.Module:
        """Return a prediction head for the decoder, cloning from encoder."""

        # Mosaic's BertForMaskedLM has the head at `cls`
        enc_head = getattr(self.encoder, "cls", None)
        if enc_head is None:
            return super()._build_decoder_head()
        return copy.deepcopy(enc_head)

    def tie_weights(self):
        """Tie projection and transformation layers between encoder and decoder heads."""
        super().tie_weights()  # Runs encoder's internal tying

        enc_lm_head = getattr(self.encoder, "cls", None)
        dec_lm_head = self.decoder_lm_head

        # Tie the final projection layer to the word embeddings
        enc_out_emb = self.encoder.get_output_embeddings()
        if (
            enc_out_emb is not None
            and enc_lm_head is not None
            and hasattr(dec_lm_head, "predictions")
            and hasattr(dec_lm_head.predictions, "decoder")
        ):
            self._tie_or_clone_weights(
                dec_lm_head.predictions.decoder, enc_lm_head.predictions.decoder
            )
        # Tie the transformation layers (dense + layernorm)
        if (
            enc_lm_head is not None
            and hasattr(enc_lm_head, "predictions")
            and hasattr(dec_lm_head, "predictions")
        ):
            enc_transform = enc_lm_head.predictions.transform
            dec_transform = dec_lm_head.predictions.transform
            self._tie_or_clone_weights(dec_transform.dense, enc_transform.dense)
            self._tie_or_clone_weights(dec_transform.LayerNorm, enc_transform.LayerNorm)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        lexmae_cfg: Dict[str, Any] | None = None,
        **hf_kwargs,
    ) -> "MosaicLexMAE":
        """
        Quick helper so users can write:
            model = BertAdapter.from_pretrained("bert-base-uncased",
                                                lexmae_cfg=dict(n_head_layers=4))
        """
        encoder: PreTrainedModel = AutoModelForMaskedLM.from_pretrained(
            model_name_or_path, trust_remote_code=True, **hf_kwargs
        )

        lexmae_cfg = lexmae_cfg or {}
        for k, v in lexmae_cfg.items():
            setattr(encoder.config, k, v)

        return cls(encoder=encoder, config=encoder.config)
