import torch.nn as nn
from transformers import (
    AutoModelForMaskedLM,
    PreTrainedModel,
)
from typing import Any, Dict
import importlib
import torch
import copy

from .lexmae_base import LexMAEBase


class _MosaicLayerWrapper(nn.Module):
    def __init__(self, core, config):
        super().__init__()
        self.core = core
        self.config = config

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
        # -----------------------------------------------------------------

        # 1) UNPAD --------------------------  [total_nnz, h]
        unpadded = hidden.view(-1, self.config.hidden_size)[flat_idx]

        bias = hidden.new_zeros(
            bs,
            self.config.num_attention_heads,
            max_len,
            max_len,
        )

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
        padded = (
            out.new_zeros(bs * L, self.config.hidden_size)
            .index_copy_(0, flat_idx, out)
            .view(bs, L, -1)
        )
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
            dec_out_emb = dec_lm_head.predictions.decoder
            if (
                hasattr(dec_out_emb, "weight")
                and enc_out_emb.weight.shape == dec_out_emb.weight.shape
            ):
                self._tie_or_clone_weights(dec_out_emb, enc_out_emb)

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
