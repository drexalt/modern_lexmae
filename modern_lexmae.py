import torch
import torch.nn as nn
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.modernbert.modeling_modernbert import (
    ModernBertForMaskedLM,
    ModernBertPredictionHead,
    ModernBertEncoderLayer,
    # ModernBertPreTrainedModel,
    _unpad_modernbert_input,
    _pad_modernbert_output,
)
from typing import Optional, Tuple, Union
from torch.nn import CrossEntropyLoss
from dataclasses import dataclass


@dataclass
class LexMAEMaskedLMOutput(MaskedLMOutput):
    sentence_embedding: Optional[torch.Tensor] = None
    dec_loss: Optional[torch.Tensor] = None
    dec_logits: Optional[torch.Tensor] = None
    dec_hidden_states: Optional[torch.Tensor] = None
    dec_attentions: Optional[torch.Tensor] = None
    enc_loss: Optional[torch.Tensor] = None
    bow_loss: Optional[torch.Tensor] = None


class ModernBertForLexMAE(ModernBertForMaskedLM):
    """
    ModernBERT model extended with LexMAE functionality, featuring dual MLM heads for encoder and decoder.
    Inherits from ModernBertForMaskedLM to leverage its encoder and adds a decoder tower.
    """

    def __init__(self, config):
        super().__init__(config)

        # Decoder-specific attributes from LexMAE
        self.n_head_layers = getattr(config, "n_head_layers", 2)
        self.skip_from = getattr(config, "skip_from", None)
        self.bow_loss_weight = getattr(config, "bow_loss_weight", 0.2)

        # Add decoder MLM head (separate from encoder's self.head)
        self.dec_head = ModernBertPredictionHead(config)

        # Add decoder layers (stack of ModernBertEncoderLayer)
        self.decoder_heads = nn.ModuleList(
            [
                ModernBertEncoderLayer(config, layer_id=i)
                for i in range(self.n_head_layers)
            ]
        )

        # Weights tied here
        self._tie_or_clone_weights(self.dec_head.dense, self.head.dense)
        self._tie_or_clone_weights(self.dec_head.norm, self.head.norm)

        self.decoder.weight.requires_grad = False
        self.decoder.bias.requires_grad = False

        k = self.n_head_layers

        # Take initialization from last layers
        for i, dec_layer in enumerate(self.decoder_heads):
            src_layer = self.model.layers[-k + i]  # encoder L‑k+i

            src_sd = src_layer.state_dict()
            tgt_sd = dec_layer.state_dict()
            filtered = {
                n: w
                for n, w in src_sd.items()
                if n in tgt_sd and w.shape == tgt_sd[n].shape
            }

            dec_layer.load_state_dict(filtered, strict=False)
        self.sparse_prediction = True

        self.special_token_ids = [self.config.cls_token_id, self.config.sep_token_id]
        # Initialize weights for new components
        self.post_init()

    ## DUP-MAE
    def ot_embedding(self, logits: torch.Tensor, attention_mask: torch.Tensor):
        """
        Project token‑level logits to a document‑level vector by
        max‑pooling over sequence positions (DupMAE Equation 3).
        Args:
            logits           – [bs, seq_len, vocab]
            attention_mask   – [bs, seq_len]  (1 = keep, 0 = padding)
        Returns:
            reps             – [bs, vocab]
        """
        mask_unsqueeze = attention_mask.unsqueeze(-1).bool()
        masked_logits = torch.where(mask_unsqueeze, logits, float("-inf"))
        reps, _ = torch.max(masked_logits, dim=1)
        return reps

    # DUP-MAE
    def bow_ot_loss(self, ot_embedding: torch.Tensor, bag_word_weight: torch.Tensor):
        """
        Cross‑entropy between pooled logits and target BoW distribution.
        Args:
            ot_embedding    – [bs, vocab]
            bag_word_weight – [bs, vocab]   (row‑normalised to 1.0)
        """
        log_probs = torch.log_softmax(ot_embedding, dim=-1)
        return torch.mean(-torch.sum(bag_word_weight * log_probs, dim=-1))

    def text_part_mask_generation(self, input_ids, special_token_ids, attention_mask):
        """
        Generate a mask for the text part, excluding special tokens.

        Args:
            input_ids: Tensor of shape [batch_size, seq_len].
            special_token_ids: List of special token IDs (e.g., [CLS], [SEP]).
            attention_mask: Tensor of shape [batch_size, seq_len].

        Returns:
            mask_text_part: Tensor of shape [batch_size, seq_len], 1 for text positions, 0 for special tokens/padding.
        """
        with torch.no_grad():
            special_token_mask = torch.isin(
                input_ids, torch.tensor(special_token_ids, device=input_ids.device)
            )
            mask_text_part = attention_mask * (~special_token_mask).long()
        return mask_text_part

    def masked_pool(self, tensor, mask, high_rank=True, method="max"):
        """
        Pool the tensor over the sequence dimension using the mask.

        Args:
            tensor: Tensor of shape [batch_size, seq_len, vocab_size].
            mask: Tensor of shape [batch_size, seq_len].
            high_rank: If True, pool over sequence dimension.
            method: Pooling method ("max" supported).

        Returns:
            pooled: Tensor of shape [batch_size, vocab_size].
        """
        if method != "max":
            raise NotImplementedError(f"Pooling method '{method}' is not implemented")
        if high_rank:
            mask_unsqueeze = mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
            tensor_masked = tensor.masked_fill(mask_unsqueeze == 0, float("-inf"))
            pooled, _ = torch.max(tensor_masked, dim=1)  # [batch_size, vocab_size]
            return pooled
        else:
            raise NotImplementedError("Only high_rank=True is implemented")

    def generate_bottleneck_repre(
        self,
        input_ids,
        attention_mask,
        bottleneck_src,
        special_token_ids,
        word_embeddings_matrix,
        last_hidden_states,
        mlm_logits,
    ):
        """
        Generate the bottleneck representation based on the specified source.

        Args:
            input_ids: Tensor of shape [batch_size, seq_len].
            attention_mask: Tensor of shape [batch_size, seq_len].
            bottleneck_src: String specifying the bottleneck method ("logits" default).
            special_token_ids: List of special token IDs.
            word_embeddings_matrix: Tensor of shape [vocab_size, hidden_size].
            last_hidden_states: Tensor of shape [batch_size, seq_len, hidden_size].
            mlm_logits: Tensor of shape [batch_size, seq_len, vocab_size].

        Returns:
            bottleneck_repre: Tensor of shape [batch_size, hidden_size].
        """
        if bottleneck_src == "cls":
            bottleneck_repre = last_hidden_states[:, 0].contiguous()
        elif bottleneck_src == "logits":
            with torch.no_grad():
                mask_text_part = self.text_part_mask_generation(
                    input_ids, special_token_ids, attention_mask
                )
            pooled_enc_logits = self.masked_pool(
                mlm_logits, mask_text_part, high_rank=True, method="max"
            )  # [bs, V]
            pooled_enc_probs = torch.softmax(pooled_enc_logits, dim=-1)  # [bs, V]
            bottleneck_repre = torch.matmul(
                pooled_enc_probs,
                word_embeddings_matrix.detach(),  # This is "gradient cut" from paper equation 4
            )  # [bs, h]
        else:
            raise NotImplementedError(
                f"Bottleneck source '{bottleneck_src}' is not implemented"
            )
        return bottleneck_repre

    def forward_decoder_heads(
        self,
        enc_cls_rep: torch.Tensor,
        dec_input_ids: Optional[torch.LongTensor] = None,
        dec_attention_mask: Optional[torch.Tensor] = None,
        dec_position_ids: Optional[torch.Tensor] = None,
        enc_hidden_states: Optional[Tuple[torch.Tensor]] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[list, Optional[list]]:
        """
        Forward pass through decoder heads, supporting both FA2 (unpadded) and non-FA2 (padded) paths.

        Args:
            enc_cls_rep: Encoder bottleneck representation [batch_size, hidden_size].
            dec_input_ids: Decoder input IDs [batch_size, seq_len], optional.
            dec_attention_mask: Decoder attention mask [batch_size, seq_len], optional.
            dec_position_ids: Decoder position IDs [batch_size, seq_len], optional.
            enc_hidden_states: Encoder hidden states tuple, optional for skip connections.
            batch_size: Batch size, inferred if not provided.
            seq_len: Sequence length, inferred if not provided.
            output_attentions: Whether to return attention weights.

        Returns:
            Tuple of (final_dec_hidden_states, dec_attentions):
            - final_dec_hidden_states: List of hidden states [batch_size, seq_len, hidden_size].
            - dec_attentions: List of attention weights if output_attentions=True, else None.
        """
        # 1. Get decoder embeddings
        if dec_input_ids is not None:
            dec_embeddings = self.model.embeddings(input_ids=dec_input_ids)
        elif self.skip_from is not None and enc_hidden_states is not None:
            dec_embeddings = enc_hidden_states[self.skip_from]
        else:
            raise ValueError(
                "Must provide dec_input_ids or enc_hidden_states with skip_from"
            )

        # 2. Infer batch_size and seq_len if not provided
        if batch_size is None or seq_len is None:
            if dec_input_ids is not None:
                batch_size, seq_len = dec_input_ids.shape[:2]
            elif dec_embeddings.dim() == 3:  # Padded embeddings
                batch_size, seq_len = dec_embeddings.shape[:2]
            else:
                raise ValueError(
                    "Need batch_size and seq_len if providing unpadded embeddings "
                    "via skip connection without input_ids"
                )

        device = dec_embeddings.device

        # 3. Prepare attention mask if not provided
        if dec_attention_mask is None:
            if dec_input_ids is not None:
                dec_attention_mask = (dec_input_ids != self.config.pad_token_id).to(
                    dtype=torch.bool, device=device
                )
            else:
                dec_attention_mask = torch.ones(
                    (batch_size, seq_len), device=device, dtype=torch.bool
                )

        # 4. Ensure dec_position_ids is available in padded format
        if dec_position_ids is None:
            dec_position_ids = (
                torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            )

        # 5. Initialize variables for FA2 and non-FA2 paths
        hidden_states = dec_embeddings  # [batch_size, seq_len, hidden_size] initially
        dec_indices = None
        dec_cu_seqlens = None
        dec_max_seqlen = None
        unpadded_dec_position_ids = None  # [total_tokens_dec] for FA2
        padded_dec_attention_mask_4d = None
        sliding_window_mask = None

        # 6. Handle FA2 unpadding
        is_unpadded = False
        if self.config._attn_implementation == "flash_attention_2":
            (
                hidden_states,
                dec_indices,
                dec_cu_seqlens,
                dec_max_seqlen,
                unpadded_dec_position_ids,
                _,
            ) = _unpad_modernbert_input(
                inputs=hidden_states,
                attention_mask=dec_attention_mask,
                position_ids=dec_position_ids,
            )
            is_unpadded = True  # hidden_states is now [total_tokens_dec, hidden_size]

        # 7. Prepare non-FA2 attention masks
        if not is_unpadded:
            padded_dec_attention_mask_4d, sliding_window_mask = (
                self._prepare_non_fa2_masks(
                    dec_attention_mask, output_attentions, batch_size, seq_len
                )
            )

        # 8. Integrate encoder bottleneck representation
        if is_unpadded:
            if enc_cls_rep.shape[0] != batch_size:
                raise ValueError(
                    f"Batch size mismatch: enc_cls_rep has batch size {enc_cls_rep.shape[0]}, "
                    f"but expected {batch_size} based on decoder input dimensions."
                )
            start_indices = dec_cu_seqlens[:-1]
            hidden_states[start_indices] = enc_cls_rep
        else:
            hidden_states[:, 0] = enc_cls_rep

        # 9. Process through decoder layers
        intermediate_dec_hidden_states = [hidden_states]
        dec_attentions = [] if output_attentions else None

        for layer in self.decoder_heads:
            layer_input = intermediate_dec_hidden_states[-1]
            if is_unpadded:
                layer_outputs = layer(
                    layer_input,
                    attention_mask=None,
                    sliding_window_mask=None,
                    position_ids=unpadded_dec_position_ids,
                    cu_seqlens=dec_cu_seqlens,
                    max_seqlen=dec_max_seqlen,
                    output_attentions=output_attentions,
                )
            else:
                layer_outputs = layer(
                    layer_input,
                    attention_mask=padded_dec_attention_mask_4d,
                    sliding_window_mask=sliding_window_mask,
                    position_ids=dec_position_ids,
                    output_attentions=output_attentions,
                )
            intermediate_dec_hidden_states.append(layer_outputs[0])
            if (
                output_attentions
                and len(layer_outputs) > 1
                and layer_outputs[1] is not None
            ):
                dec_attentions.append(layer_outputs[1])

        # 10. Repad hidden states if unpadded
        final_dec_hidden_states = (
            self._repad_hidden_states(
                intermediate_dec_hidden_states, dec_indices, batch_size, seq_len
            )
            if is_unpadded
            else intermediate_dec_hidden_states
        )

        # 11. Finalize attentions
        dec_attentions = (
            dec_attentions if output_attentions and dec_attentions else None
        )

        return final_dec_hidden_states, dec_attentions

    def _prepare_non_fa2_masks(
        self, attention_mask, output_attentions, batch_size, seq_len
    ):
        """
        Prepare 4D attention mask and sliding window mask for non-FA2 path.

        Args:
            attention_mask: 2D attention mask [batch_size, seq_len].
            output_attentions: Whether to compute attention weights.
            batch_size: Batch size.
            seq_len: Sequence length.

        Returns:
            Tuple of (mask_4d, sliding_mask).
        """
        mask_4d, sliding_mask = self.model._update_attention_mask(
            attention_mask, output_attentions=output_attentions
        )
        return mask_4d, sliding_mask

    def _repad_hidden_states(self, hidden_states_list, indices, batch_size, seq_len):
        """
        Repad hidden states from unpadded to padded format.

        Args:
            hidden_states_list: List of unpadded hidden states [total_tokens_dec, hidden_size].
            indices: Indices for repadding.
            batch_size: Target batch size.
            seq_len: Target sequence length.

        Returns:
            List of padded hidden states [batch_size, seq_len, hidden_size].
        """
        return [
            _pad_modernbert_output(hs, indices, batch_size, seq_len)
            for hs in hidden_states_list
        ]

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sliding_window_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,  # Encoder MLM labels
        dec_input_ids: Optional[torch.LongTensor] = None,
        dec_attention_mask: Optional[torch.Tensor] = None,
        dec_position_ids: Optional[torch.Tensor] = None,
        dec_labels: Optional[torch.Tensor] = None,  # Decoder MLM labels
        enc_cls_rep: Optional[torch.Tensor] = None,
        enc_hidden_states: Optional[Tuple[torch.Tensor]] = None,
        indices: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        disable_encoding: bool = False,
        disable_decoding: bool = True,
        bag_word_weight: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, LexMAEMaskedLMOutput]:
        """
        Forward pass for ModernBertForLexMAE with dual MLM tasks.

        Args:
            Same as ModernBertForMaskedLM, plus:
            dec_input_ids, dec_attention_mask, dec_position_ids: Decoder inputs
            dec_labels: Decoder MLM labels
            enc_cls_rep, enc_hidden_states: Precomputed encoder outputs
            disable_encoding: Skip encoder if True
            disable_decoding: Skip decoder if True
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        self._maybe_set_compile()

        return_dict_out = LexMAEMaskedLMOutput()
        input_ids_padded = input_ids.clone() if input_ids is not None else None
        labels_padded = labels.clone() if labels is not None else None
        # Encoder Phase
        if not disable_encoding:
            # Unpad inputs if using Flash Attention
            if (
                self.config._attn_implementation == "flash_attention_2"
                and indices is None
            ):
                if batch_size is None or seq_len is None:
                    if inputs_embeds is not None:
                        batch_size, seq_len = inputs_embeds.shape[:2]
                    elif input_ids is not None:
                        batch_size, seq_len = input_ids.shape[:2]
                    else:
                        raise ValueError(
                            "Either input_ids or inputs_embeds must be provided"
                        )
                device = (
                    input_ids.device if input_ids is not None else inputs_embeds.device
                )
                if attention_mask is None:
                    attention_mask = torch.ones(
                        (batch_size, seq_len), device=device, dtype=torch.bool
                    )
                if inputs_embeds is not None:
                    with torch.no_grad():
                        (
                            inputs_embeds,
                            indices,
                            cu_seqlens,
                            max_seqlen,
                            position_ids,
                            labels,
                        ) = _unpad_modernbert_input(
                            inputs=inputs_embeds,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            labels=labels,
                        )
                else:
                    with torch.no_grad():
                        (
                            input_ids,
                            indices,
                            cu_seqlens,
                            max_seqlen,
                            position_ids,
                            labels,
                        ) = _unpad_modernbert_input(
                            inputs=input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            labels=labels,
                        )

            # Run encoder
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                sliding_window_mask=sliding_window_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                indices=indices,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                batch_size=batch_size,
                seq_len=seq_len,
                output_attentions=output_attentions,
                output_hidden_states=True,  # Always get hidden states for LexMAE
                return_dict=True,
            )
            last_hidden_state = outputs.last_hidden_state
            # Repad last_hidden_state if using Flash Attention
            if self.config._attn_implementation == "flash_attention_2":
                with torch.no_grad():
                    last_hidden_state = _pad_modernbert_output(
                        last_hidden_state, indices, batch_size, seq_len
                    )  # [4, 512, hidden_size]

            # Compute MLM logits from padded last_hidden_state for bottleneck
            enc_logits_full = (
                self.compiled_head(last_hidden_state)
                if self.config.reference_compile
                else self.decoder(self.head(last_hidden_state))
            )

            enc_loss = None
            if labels_padded is not None:
                if self.sparse_prediction:
                    labels_flat = labels_padded.view(-1)
                    mask_tokens = labels_flat != self.sparse_pred_ignore_index
                    enc_logits = enc_logits_full.view(-1, self.config.vocab_size)[
                        mask_tokens
                    ]
                    labels_loss = labels_flat[mask_tokens]
                else:
                    enc_logits = enc_logits_full
                    labels_loss = labels_padded
                enc_loss = CrossEntropyLoss()(
                    enc_logits.view(-1, self.config.vocab_size), labels_loss.view(-1)
                )

            bow_loss = None
            if bag_word_weight is not None:
                mask_text_part = self.text_part_mask_generation(
                    input_ids_padded, self.special_token_ids, attention_mask
                )
                # skip [CLS] so shape matches Dup‑MAE impl
                ot_emb = self.ot_embedding(enc_logits_full, mask_text_part)  # [bs, V]
                bow_loss = self.bow_ot_loss(ot_emb, bag_word_weight)

            # Generate bottleneck representation with padded mlm_logits
            enc_cls_rep = self.generate_bottleneck_repre(
                input_ids=input_ids_padded,
                attention_mask=attention_mask,
                bottleneck_src="logits",
                special_token_ids=self.special_token_ids,
                word_embeddings_matrix=self.model.embeddings.tok_embeddings.weight,
                last_hidden_states=last_hidden_state,
                mlm_logits=enc_logits_full,  # Now [4, 512, vocab_size]
            )

            # # Compute encoder loss with sparse prediction if applicable
            # print(f"labels shape: {labels.shape}")
            # if self.sparse_prediction and labels is not None:
            #     labels_flat = labels.view(-1)
            #     mask_tokens = labels_flat != self.sparse_pred_ignore_index
            #     enc_logits = enc_logits_full.view(-1, self.config.vocab_size)[
            #         mask_tokens
            #     ]
            #     labels = labels_flat[mask_tokens]
            # else:
            #     enc_logits = enc_logits_full  # Keep full shape for consistency

            # enc_loss = None
            # print(f"enc_logits: {enc_logits.shape}")
            # print(f"labels: {labels.shape}")
            # if labels is not None:
            #     enc_loss = CrossEntropyLoss()(
            #         enc_logits.view(-1, self.config.vocab_size), labels.view(-1)
            #     )

            # Repad encoder outputs if using Flash Attention
            # if self.config._attn_implementation == "flash_attention_2":
            #     with torch.no_grad():
            #         enc_logits = _pad_modernbert_output(
            #             enc_logits, indices, batch_size, seq_len
            #         )
            # last_hidden_state = _pad_modernbert_output(
            #     last_hidden_state, indices, batch_size, seq_len
            # )
            enc_hidden_states = outputs.hidden_states

            # Prepare return dictionary
            return_dict_out.enc_loss = enc_loss
            return_dict_out.bow_loss = bow_loss
            return_dict_out.logits = enc_logits_full
            return_dict_out.sentence_embedding = enc_cls_rep
            return_dict_out.hidden_states = enc_hidden_states
            return_dict_out.attentions = outputs.attentions

            if enc_loss is not None or bow_loss is not None:
                enc_loss = (
                    enc_loss
                    if enc_loss is not None
                    else torch.tensor(0.0, device=enc_logits_full.device)
                )
                bow_loss = (
                    bow_loss
                    if bow_loss is not None
                    else torch.tensor(0.0, device=enc_logits_full.device)
                )
                return_dict_out.loss = enc_loss + bow_loss

        else:
            if enc_cls_rep is None or enc_hidden_states is None:
                raise ValueError(
                    "Must provide enc_cls_rep and enc_hidden_states when disable_encoding=True"
                )

        # Decoder Phase
        if not disable_decoding:
            if dec_input_ids is not None:
                dec_batch_size, dec_seq_len = dec_input_ids.shape[:2]
            elif enc_hidden_states is not None and self.skip_from is not None:
                dec_batch_size, dec_seq_len = enc_hidden_states[self.skip_from].shape[
                    :2
                ]
            else:
                raise ValueError(
                    "Must provide dec_input_ids or enc_hidden_states with skip_from for decoding"
                )

            dec_hidden_states, dec_attentions = self.forward_decoder_heads(
                enc_cls_rep,
                dec_input_ids=dec_input_ids,
                dec_attention_mask=dec_attention_mask,
                dec_position_ids=dec_position_ids,
                enc_hidden_states=enc_hidden_states,
                # indices=indices,
                # cu_seqlens=cu_seqlens,
                # max_seqlen=max_seqlen,
                batch_size=dec_batch_size,
                seq_len=dec_seq_len,
                output_attentions=output_attentions,
            )

            # Decoder MLM prediction
            dec_last_hidden = dec_hidden_states[-1]
            if self.sparse_prediction and dec_labels is not None:
                dec_labels_flat = dec_labels.view(-1)
                dec_mask_tokens = dec_labels_flat != self.sparse_pred_ignore_index
                masked_indices = torch.where(dec_mask_tokens)[0]
                dec_last_hidden_flat = dec_last_hidden.view(
                    -1, dec_last_hidden.size(-1)
                )
                dec_last_hidden_masked = dec_last_hidden_flat[dec_mask_tokens]
                dec_logits = (
                    self.compiled_head(dec_last_hidden_masked)
                    if self.config.reference_compile
                    else self.decoder(self.dec_head(dec_last_hidden_masked))
                )
                dec_labels_masked = dec_labels_flat[dec_mask_tokens]
            else:
                dec_logits = (
                    self.compiled_head(dec_last_hidden)
                    if self.config.reference_compile
                    else self.decoder(self.dec_head(dec_last_hidden))
                )
                dec_labels_masked = dec_labels

            # Compute decoder loss
            dec_loss = None
            if dec_labels is not None:
                dec_loss = CrossEntropyLoss()(
                    dec_logits.view(-1, self.config.vocab_size),
                    dec_labels_masked.view(-1),
                )
            # # Repad decoder logits - this is not correct, masked_indices should be dec_indices, but we're not using logits anyway
            # if self.config._attn_implementation == "flash_attention_2":
            #     with torch.no_grad():
            #         dec_logits = _pad_modernbert_output(
            #             dec_logits, masked_indices, dec_batch_size, dec_seq_len
            #         )

            # Update return dictionary
            return_dict_out.dec_loss = dec_loss
            return_dict_out.dec_logits = dec_logits
            return_dict_out.dec_hidden_states = dec_hidden_states
            return_dict_out.dec_attentions = dec_attentions

        if not return_dict:
            outputs = (
                return_dict_out.logits,
                return_dict_out.dec_logits if not disable_decoding else None,
            )
            return (
                ((return_dict_out.loss, return_dict_out.dec_loss) + outputs)
                if return_dict_out.loss
                else outputs
            )

        return return_dict_out

    def _init_weights(self, module):
        """Override weight initialization for new components."""
        super()._init_weights(module)
        if isinstance(module, ModernBertPredictionHead) and module is self.dec_head:
            self._init_weights(module.dense)
        elif (
            isinstance(module, ModernBertEncoderLayer) and module in self.decoder_heads
        ):
            self._init_weights(module.attn.Wqkv)
            self._init_weights(module.attn.Wo)
            self._init_weights(module.mlp.Wi)
            self._init_weights(module.mlp.Wo)
