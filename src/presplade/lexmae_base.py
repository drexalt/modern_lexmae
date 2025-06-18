from dataclasses import dataclass
from typing import Optional, Tuple, Union, List, Dict, Any

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput

from .utils import (
    generate_bottleneck_repre,
    ot_embedding,
    bow_ot_loss,
    text_part_mask_generation,
)


@dataclass
class LexMAEMaskedLMOutput(MaskedLMOutput):
    """Return type for :class:`LexMAEBase.forward`. All fields are optional so
    subclasses can populate only what they use.
    """

    sentence_embedding: Optional[torch.Tensor] = None  # [bs, hidden]
    # Decoder‑side artefacts
    dec_loss: Optional[torch.Tensor] = None
    dec_logits: Optional[torch.Tensor] = None  # [bs, L, V] or sparse
    dec_hidden_states: Optional[Tuple[torch.Tensor]] = None
    dec_attentions: Optional[Tuple[torch.Tensor]] = None
    # Extra losses (DupMAE BoW)
    bow_loss: Optional[torch.Tensor] = None
    enc_loss: Optional[torch.Tensor] = None  # alias for MaskedLMOutput.loss


class LexMAEBase(PreTrainedModel):
    """Model‑agnostic LexMAE scaffold.

    Parameters
    ----------
    encoder : transformers.PreTrainedModel
        A **masked‑LM** encoder that already has its own MLM head.  Must expose
        `config.hidden_size`, `config.vocab_size`, `.get_input_embeddings()` and
        `.get_output_embeddings()` (all Hugging Face masked‑LM models do).
    config  : transformers.PretrainedConfig (typically LexMAEConfig)
        Configuration object that *extends* the underlying encoder config with
        LexMAE‑specific hyper‑parameters (``n_head_layers``, ``skip_from`` …).
    """

    def _build_decoder_layer(self) -> nn.Module:  # pragma: no cover
        """Return a **single** decoder layer compatible with the encoder.
        Must be overridden in adapter subclasses.
        """

        raise NotImplementedError("Adapter must implement _build_decoder_layer()")

    def _build_decoder_head(self) -> nn.Module:  # pragma: no cover
        """Return a prediction head (hidden → vocab) for the decoder tower.
        The default implementation *clones* the encoder's MLM head if possible.
        Override if your encoder has a non‑standard head (e.g. ELECTRA).
        """

        enc_head = self.encoder.lm_head if hasattr(self.encoder, "lm_head") else None
        if enc_head is None:  # Fallback – just a linear projection tied to emb
            hidden_size = self.encoder.config.hidden_size
            vocab_size = self.encoder.config.vocab_size
            head = nn.Linear(hidden_size, vocab_size, bias=False)
            head.weight = self.encoder.get_output_embeddings().weight  # weight tying
            return head
        # Deep‑copy to avoid parameter aliasing; we tie later in tie_weights()
        import copy

        return copy.deepcopy(enc_head)

    def cls_sep_ids(self) -> Tuple[int, int]:  # pragma: no cover
        """Return ``(cls_id, sep_id)`` for the *wrapped encoder's* tokenizer.
        Subclasses should override **only** if the encoder config does not carry
        these fields (rare).
        """

        cls_id = getattr(self.encoder.config, "cls_token_id", 0)
        sep_id = getattr(self.encoder.config, "sep_token_id", 0)
        return cls_id, sep_id

    def __init__(self, encoder: PreTrainedModel, config):  # noqa: D401
        super().__init__(config)
        self.encoder: PreTrainedModel = encoder  # composed, not inherited

        # LexMAE‑specific hyper‑parameters (with sane defaults)
        self.n_head_layers: int = getattr(config, "n_head_layers", 2)
        self.skip_from: Optional[int] = getattr(config, "skip_from", None)
        self.bow_loss_weight: float = getattr(config, "bow_loss_weight", 0.0)
        self.bottleneck_src: str = getattr(config, "bottleneck_src", "logits")

        # Special tokens for (a) bottleneck masking, (b) DupMAE BoW loss
        self.special_token_ids: List[int] = list(self.cls_sep_ids())

        # ---- Decoder tower -------------------------------------------------
        self.decoder_heads = nn.ModuleList(
            [self._build_decoder_layer() for _ in range(self.n_head_layers)]
        )
        self.decoder_lm_head = self._build_decoder_head()

        # HF housekeeping – tie/clone embeddings *after* everything exists
        self.tie_weights()

    def get_output_embeddings(self):
        """Return *encoder* output embeddings (for HF generation utils)."""

        return self.encoder.get_output_embeddings()

    def tie_weights(self):  # noqa: D401
        """Share weights encoder↔decoder when shapes allow – mimics HF logic."""

        # 1) call encoder's own tying routine first (if any)
        if hasattr(self.encoder, "tie_weights"):
            self.encoder.tie_weights()

        # 2) tie decoder's projection matrix to encoder's word embeddings
        enc_out_emb = self.encoder.get_output_embeddings()
        dec_out_emb = (
            self.decoder_lm_head.decoder
            if hasattr(self.decoder_lm_head, "decoder")  # BERT/Roberta style
            else self.decoder_lm_head
        )
        if (
            enc_out_emb is not None
            and dec_out_emb.weight.shape == enc_out_emb.weight.shape
        ):
            self._tie_or_clone_weights(dec_out_emb, enc_out_emb)

    def forward_decoder_heads(
        self,
        cls_rep: torch.Tensor,
        dec_embeddings: torch.Tensor,
        dec_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]]]:
        """Run the mini‑transformer that reconstructs tokens from the
        *bottleneck* representation + (optionally) skip‑connected embeddings.
        „dec_embeddings” is expected to be **padded**.
        """

        # Replace [CLS] (pos=0) by the sentence‑level bottleneck representation
        dec_init_state = torch.cat([cls_rep.unsqueeze(1), dec_embeddings[:, 1:]], dim=1)

        hidden_states: List[torch.Tensor] = [dec_init_state]
        attentions: List[torch.Tensor] | None = [] if output_attentions else None

        for layer in self.decoder_heads:
            out = layer(
                hidden_states[-1],
                attention_mask=dec_attention_mask,
                output_attentions=output_attentions,
            )
            hidden_states.append(out[0])
            if output_attentions:
                attentions.append(out[1])

        return hidden_states, attentions if output_attentions else None

    def forward(
        self,
        # Encoder inputs
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        enc_mlm_labels: Optional[torch.Tensor] = None,
        bag_word_weight: Optional[torch.Tensor] = None,
        # Decoder inputs
        dec_input_ids: Optional[torch.LongTensor] = None,
        dec_attention_mask: Optional[torch.Tensor] = None,
        dec_position_ids: Optional[torch.Tensor] = None,
        dec_mlm_labels: Optional[torch.Tensor] = None,
        # Control flags
        disable_encoding: bool = False,
        disable_decoding: bool = True,
        output_attentions: bool = False,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Union[LexMAEMaskedLMOutput, Tuple]:  # noqa: D401
        """High‑level forward orchestrating encoder → bottleneck → decoder."""

        # -------------------------------------
        # 1) Encoder phase (MLM + bottleneck rep)
        # -------------------------------------
        if not disable_encoding:
            enc_out: MaskedLMOutput = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                labels=enc_mlm_labels,
                output_hidden_states=True,
                return_dict=True,
            )
            encoder_logits = enc_out.logits  # [bs, L, V]
            encoder_hidden = enc_out.hidden_states  # tuple(len_layers) of [bs, L, h]

            last_hidden = encoder_hidden[-1] if encoder_hidden is not None else None

            # Sentence‑level bottleneck representation (Eq. 4)
            cls_rep = generate_bottleneck_repre(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bottleneck_src=self.bottleneck_src,
                special_token_ids=self.special_token_ids,
                word_embeddings_matrix=self.encoder.get_input_embeddings().weight,
                last_hidden_states=last_hidden,
                mlm_logits=encoder_logits,
            )  # [bs, h]

            # BoW loss (Dup‑MAE) – optional
            bow_loss = None
            if bag_word_weight is not None:
                mask_text = text_part_mask_generation(
                    input_ids, self.special_token_ids, attention_mask
                )
                ot_emb = ot_embedding(encoder_logits, mask_text)
                bow_loss = bow_ot_loss(ot_emb, bag_word_weight)

            enc_loss = enc_out.loss  # already computed by HF loss head
        else:
            assert "enc_cls_rep" in kwargs and "enc_hidden_states" in kwargs, (
                "When disable_encoding=True you must pass pre‑computed \n"
                " enc_cls_rep and enc_hidden_states via kwargs."
            )
            cls_rep = kwargs["enc_cls_rep"]
            encoder_hidden = kwargs["enc_hidden_states"]
            bow_loss = None
            enc_loss = None

        dec_loss = dec_logits = dec_hidden = dec_attn = None
        if not disable_decoding:
            # 2a) Prepare embeddings for the decoder tower
            if dec_input_ids is not None:
                dec_embeddings = self.encoder.get_input_embeddings()(dec_input_ids)
            else:
                assert self.skip_from is not None, (
                    "Need skip_from index to grab embeddings from encoder"
                )
                dec_embeddings = encoder_hidden[self.skip_from]

            # 2b) Fallback masks
            dec_attention_mask = (
                dec_attention_mask if dec_attention_mask is not None else attention_mask
            )

            # 2c) Run decoder stack
            dec_hidden, dec_attn = self.forward_decoder_heads(
                cls_rep,
                dec_embeddings=dec_embeddings,
                dec_attention_mask=dec_attention_mask,
                output_attentions=output_attentions,
            )
            dec_last = dec_hidden[-1]
            dec_logits = self.decoder_lm_head(dec_last)

            if dec_mlm_labels is not None:
                dec_loss = CrossEntropyLoss()(
                    dec_logits.view(-1, self.encoder.config.vocab_size),
                    dec_mlm_labels.view(-1),
                )

        if not return_dict:
            # Classic tuple order mirrors HF MaskedLMOutput order + extras
            output: List[Any] = [
                enc_loss,
                encoder_logits,
                encoder_hidden,
            ]
            if not disable_decoding:
                output.extend([dec_loss, dec_logits, dec_hidden, dec_attn])
            return tuple(output)

        return LexMAEMaskedLMOutput(
            loss=enc_loss,  # alias for encoder MLM loss
            enc_loss=enc_loss,
            bow_loss=bow_loss,
            logits=encoder_logits,
            hidden_states=encoder_hidden,
            sentence_embedding=cls_rep,
            # decoder‑side
            dec_loss=dec_loss,
            dec_logits=dec_logits,
            dec_hidden_states=dec_hidden,
            dec_attentions=dec_attn,
        )
