# pyright: basic

"""
NeoBERT ↔ LexMAE adapter
------------------------

Compose a NeoBERT masked‑LM encoder with a small decoder tower for LexMAE.
We grab the exact `EncoderBlock` symbol from the loaded model’s module to
build decoder layers — no fallbacks or name‑search heuristics.

Only the number of decoder layers differs; all other parameters mirror the
encoder. Weight tying for heads is handled by LexMAEBase defaults.
"""

from __future__ import annotations

from typing import Any, Dict
import torch.nn as nn
from transformers import AutoModelForMaskedLM, PreTrainedModel
import importlib
import types

from .lexmae_base import LexMAEBase


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
        # Import EncoderBlock symbol from the loaded model’s module (like Mosaic path)
        module = importlib.import_module(self.encoder.__class__.__module__)
        if not hasattr(module, "EncoderBlock"):
            raise ImportError(
                "Expected `EncoderBlock` in the loaded NeoBERT module. "
                "Ensure your model defines `EncoderBlock(config)` in the same module as the LM head."
            )
        Block = getattr(module, "EncoderBlock")

        class _EncoderBlockWrapper(nn.Module):
            def __init__(self, config, freqs_getter):
                super().__init__()
                self.block = Block(config)
                self.config = config
                self._freqs_getter = freqs_getter

            def forward(
                self, hidden_states, attention_mask=None, output_attentions=False
            ):
                bs, seqlen, _ = hidden_states.shape
                # Expand 2D mask → [bs, heads, L, L] to match EncoderBlock multiply path
                attn_mask = None
                if attention_mask is not None:
                    if attention_mask.dim() == 2:
                        attn_mask = (
                            attention_mask.unsqueeze(1)
                            .unsqueeze(1)
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
                return (out, attn) if output_attentions else (out,)

        return _EncoderBlockWrapper(self.encoder.config, self._get_freqs_cis)

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
        # 1) Load a masked‑LM checkpoint (trust remote code by default)
        if "trust_remote_code" not in hf_kwargs:
            hf_kwargs["trust_remote_code"] = True
        encoder: PreTrainedModel = AutoModelForMaskedLM.from_pretrained(
            model_name_or_path, **hf_kwargs
        )

        # 2) Extend config with LexMAE fields
        lexmae_cfg = lexmae_cfg or {}
        for k, v in lexmae_cfg.items():
            setattr(encoder.config, k, v)

        # 2b) Ensure HF embedding accessors exist and work on this instance.
        #     Override even if base class defines stubs that return None.
        if not hasattr(encoder, "model") or not hasattr(encoder.model, "encoder"):
            raise AttributeError(
                "Loaded NeoBERT LM head must expose `model.encoder` (input embeddings)."
            )
        if not hasattr(encoder, "decoder"):
            raise AttributeError(
                "Loaded NeoBERT LM head must expose `decoder` (output embeddings / LM head)."
            )

        def _get_input_embeddings(self):
            return self.model.encoder  # type: ignore[attr-defined]

        def _set_input_embeddings(self, value):
            self.model.encoder = value  # type: ignore[attr-defined]

        def _get_output_embeddings(self):
            return self.decoder  # type: ignore[attr-defined]

        def _set_output_embeddings(self, value):
            self.decoder = value  # type: ignore[attr-defined]

        encoder.get_input_embeddings = types.MethodType(
            _get_input_embeddings, encoder
        )  # type: ignore[attr-defined]
        encoder.set_input_embeddings = types.MethodType(
            _set_input_embeddings, encoder
        )  # type: ignore[attr-defined]
        encoder.get_output_embeddings = types.MethodType(
            _get_output_embeddings, encoder
        )  # type: ignore[attr-defined]
        encoder.set_output_embeddings = types.MethodType(
            _set_output_embeddings, encoder
        )  # type: ignore[attr-defined]

        # 3) Wrap and return
        return cls(encoder=encoder, config=encoder.config)
