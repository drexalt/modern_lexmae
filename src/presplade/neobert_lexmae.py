"""
NeoBERT ↔ LexMAE adapter
------------------------

This mirrors the working BERT-based LexMAE path and composes a
`transformers` masked‑LM encoder with a small decoder tower for LexMAE.

Notes
- Requires a NeoBERT decoder/encoder layer class to be importable. No fallbacks
  to BERT blocks are used.
- Weight tying for the decoder head is handled by LexMAEBase’s default
  logic (clone/tie to the encoder MLM head when shapes permit).
"""

from __future__ import annotations

from typing import Any, Dict
import torch.nn as nn
from transformers import AutoModelForMaskedLM, PreTrainedModel

from .lexmae_base import LexMAEBase


"""Resolve a NeoBERT layer class from likely locations.

We accept any of the following (first found wins):
  - transformers.models.neobert.modeling_neobert.NeoBertEncoderLayer
  - transformers.models.neobert.modeling_neobert.NeoBertLayer
  - transformers.models.neobert.modeling_neobert.EncoderBlock (custom impl)
"""

NeoBertLayerType = None
_NEOBERT_LAYER_NAME = None
_layer_import_error = None
try:  # common naming in HF-style ports
    from transformers.models.neobert.modeling_neobert import (  # type: ignore
        NeoBertEncoderLayer as _NeoBertLayer,
    )

    NeoBertLayerType = _NeoBertLayer
    _NEOBERT_LAYER_NAME = getattr(_NeoBertLayer, "__name__", "NeoBertEncoderLayer")
except Exception as e1:  # pragma: no cover
    try:
        from transformers.models.neobert.modeling_neobert import (  # type: ignore
            NeoBertLayer as _NeoBertLayer,
        )

        NeoBertLayerType = _NeoBertLayer
        _NEOBERT_LAYER_NAME = getattr(_NeoBertLayer, "__name__", "NeoBertLayer")
    except Exception as e2:  # pragma: no cover
        try:
            from transformers.models.neobert.modeling_neobert import (  # type: ignore
                EncoderBlock as _NeoBertLayer,
            )

            NeoBertLayerType = _NeoBertLayer
            _NEOBERT_LAYER_NAME = getattr(_NeoBertLayer, "__name__", "EncoderBlock")
        except Exception as e3:  # pragma: no cover
            _layer_import_error = (e1, e2, e3)


class NeoBertAdapter(LexMAEBase):
    """LexMAE adapter for NeoBERT-like encoders.

    It composes a Hugging Face masked‑LM encoder (loaded via
    `AutoModelForMaskedLM.from_pretrained`) and builds a decoder tower out of
    NeoBERT encoder layers. No BERT fallbacks are used.
    """

    def _get_freqs_cis(self, device, seqlen):
        try:
            freqs = self.encoder.model.freqs_cis  # type: ignore[attr-defined]
        except AttributeError as e:
            raise AttributeError(
                "NeoBERTLMHead must expose `model.freqs_cis` buffer to drive rotary embeddings "
                "in decoder layers, or adapt the adapter to your implementation."
            ) from e
        return freqs[:seqlen].unsqueeze(0).to(device)

    def _build_decoder_layer(self) -> nn.Module:
        if NeoBertLayerType is None:
            raise ImportError(
                "Could not locate a NeoBERT layer class. Expected one of: "
                "NeoBertEncoderLayer, NeoBertLayer, or EncoderBlock in "
                "transformers.models.neobert.modeling_neobert. Ensure your NeoBERT "
                "package/registers these symbols."
            )
        # If we resolved to a custom EncoderBlock, wrap it to match the
        # (hidden, attn) → tuple API used by LexMAEBase.
        if _NEOBERT_LAYER_NAME == "EncoderBlock":
            Block = NeoBertLayerType  # type: ignore[assignment]

            class _EncoderBlockWrapper(nn.Module):
                def __init__(self, config, freqs_getter):
                    super().__init__()
                    self.block = Block(config)
                    self.config = config
                    self._freqs_getter = freqs_getter

                def forward(self, hidden_states, attention_mask=None, output_attentions=False):
                    bs, seqlen, _ = hidden_states.shape
                    # Expand attention mask to [bs, heads, L, L] if provided
                    attn_mask = None
                    if attention_mask is not None:
                        if attention_mask.dim() == 2:
                            attn_mask = (
                                attention_mask.unsqueeze(1).unsqueeze(1)
                                .repeat(1, self.config.num_attention_heads, seqlen, 1)
                            )
                        else:
                            attn_mask = attention_mask
                    freqs = self._freqs_getter(hidden_states.device, seqlen)
                    out, attn = self.block(
                        hidden_states,
                        attn_mask,
                        freqs,
                        output_attentions,
                        max_seqlen=None,
                        cu_seqlens=None,
                    )
                    if output_attentions:
                        return (out, attn)
                    return (out,)

            return _EncoderBlockWrapper(self.encoder.config, self._get_freqs_cis)

        # Otherwise, assume HF-style layer signature
        return NeoBertLayerType(self.encoder.config)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        lexmae_cfg: Dict[str, Any] | None = None,
        **hf_kwargs,
    ) -> "NeoBertAdapter":
        """Load a masked‑LM encoder and wrap it as a NeoBERT LexMAE model.

        Example
        -------
        >>> model = NeoBertAdapter.from_pretrained(
        ...     "<hf-org-or-user>/<neobert-id>",
        ...     lexmae_cfg=dict(n_head_layers=2, skip_from=-2, bottleneck_src="logits"),
        ... )
        """
        # 1) Load a regular masked‑LM checkpoint from HF Hub
        encoder: PreTrainedModel = AutoModelForMaskedLM.from_pretrained(
            model_name_or_path, **hf_kwargs
        )

        # 2) Extend config with LexMAE fields
        lexmae_cfg = lexmae_cfg or {}
        for k, v in lexmae_cfg.items():
            setattr(encoder.config, k, v)

        # 3) Wrap and return
        return cls(encoder=encoder, config=encoder.config)
