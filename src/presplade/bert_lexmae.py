from typing import Any, Dict

import torch.nn as nn
from transformers import (
    AutoModelForMaskedLM,
    PreTrainedModel,
)
from transformers.models.bert.modeling_bert import BertLayer

from .lexmae_base import LexMAEBase


class BertAdapter(LexMAEBase):
    """
    The only thing we have to teach LexMAEBase about BERT is how to
    build **one** transformer block for the decoder tower.
    """

    def _build_decoder_layer(self) -> nn.Module:
        """Return a single `BertLayer` initialised with the encoder config."""
        return BertLayer(self.encoder.config)

    def tie_weights(self):
        """Tie prediction head weights in addition to base class tying."""
        super().tie_weights()

        enc_lm_head = self.encoder.get_lm_head()
        dec_lm_head = self.decoder_lm_head

        if hasattr(enc_lm_head, "transform") and hasattr(dec_lm_head, "transform"):
            enc_transform = enc_lm_head.transform
            dec_transform = dec_lm_head.transform
            self._tie_or_clone_weights(dec_transform.dense, enc_transform.dense)
            self._tie_or_clone_weights(dec_transform.LayerNorm, enc_transform.LayerNorm)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        lexmae_cfg: Dict[str, Any] | None = None,
        **hf_kwargs,
    ) -> "BertAdapter":
        """
        Quick helper so users can write:
            model = BertAdapter.from_pretrained("bert-base-uncased",
                                                lexmae_cfg=dict(n_head_layers=4))
        """
        # 1) load a regular *masked-LM* checkpoint from HF Hub
        encoder: PreTrainedModel = AutoModelForMaskedLM.from_pretrained(
            model_name_or_path, **hf_kwargs
        )

        lexmae_cfg = lexmae_cfg or {}
        for k, v in lexmae_cfg.items():
            setattr(encoder.config, k, v)

        return cls(encoder=encoder, config=encoder.config)
