import torch
from functools import reduce
import os
from heapq import heappush, heapreplace, heappop


def save_checkpoint(
    step: int,
    loss: float,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
) -> str:
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "loss": loss,
    }
    filepath = os.path.join(
        checkpoint_path, f"checkpoint_step_{step}_loss_{loss:.4f}.pt"
    )
    torch.save(checkpoint, filepath)
    return filepath


def update_checkpoint_tracking(
    step: int,
    loss: float,
    checkpoint_losses: list,
    max_checkpoints: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
) -> list:
    updated_checkpoint_losses = checkpoint_losses.copy()
    neg_loss = -loss

    if len(updated_checkpoint_losses) < max_checkpoints:
        filepath = save_checkpoint(step, loss, model, optimizer, checkpoint_path)
        heappush(updated_checkpoint_losses, (neg_loss, step, filepath))
    elif neg_loss > updated_checkpoint_losses[0][0]:
        new_filepath = save_checkpoint(step, loss, model, optimizer, checkpoint_path)
        _, old_step, old_filepath = heapreplace(
            updated_checkpoint_losses, (neg_loss, step, new_filepath)
        )
        print(f"Checkpoint saved at {new_filepath}")
        if os.path.exists(old_filepath):
            try:
                os.remove(old_filepath)
            except OSError as e:
                print(f"Error removing old checkpoint {old_filepath}: {e}")

    return updated_checkpoint_losses


def special_token_mask_generation(input_ids, special_token_ids):
    init_no_mask = torch.full_like(input_ids, False, dtype=torch.bool)
    mask_bl = reduce(
        lambda acc, el: acc | (input_ids == el), special_token_ids, init_no_mask
    )
    return mask_bl.to(torch.long)


def text_part_mask_generation(input_ids, special_token_ids, attention_mask):
    mask_text_part = (
        1 - special_token_mask_generation(input_ids, special_token_ids)
    ) * attention_mask
    return mask_text_part


def mlm_input_ids_masking_onthefly(
    input_ids,
    attention_mask,
    mask_token_id,
    special_token_ids,
    mlm_prob=0.15,
    mlm_rdm_prob=0.1,
    mlm_keep_prob=0.1,
    vocab_size=None,
    external_rdm_probs=None,
    extra_masked_flags=None,
    exclude_flags=None,
    resample_nonmask=False,
):
    """
    Masks input token IDs for masked language modeling (MLM) on-the-fly.

    Parameters:
    -----------
    input_ids : torch.Tensor
        Input sequence token IDs.
    attention_mask : torch.Tensor
        Padding token mask.
    mask_token_id : int
        [MASK] token ID.
    special_token_ids : list or torch.Tensor
        Special token IDs.
    mlm_prob : float, optional, default=0.15
        Masking probability.
    mlm_rdm_prob : float, optional, default=0.1
        Random replacement probability.
    mlm_keep_prob : float, optional, default=0.1
        Keep unchanged probability.
    vocab_size : int, optional
        Vocabulary size.
    external_rdm_probs : torch.Tensor, optional
        Custom masking probabilities.
    extra_masked_flags : torch.Tensor, optional
        Extra tokens to mask.
    exclude_flags : torch.Tensor, optional
        Tokens to exclude.
    resample_nonmask : bool, optional, default=False
        New replacement randomness.

    Returns:
    --------
    masked_input_ids : torch.Tensor
        Masked input sequence.
    selected_for_masking : torch.Tensor
        Selected token flags.
    mlm_labels : torch.Tensor
        Original masked tokens.
    random_probs : torch.Tensor
        Used random probabilities.
    """
    with torch.no_grad():
        if external_rdm_probs is None:
            random_probs = torch.rand(input_ids.shape, device=input_ids.device)
        else:
            random_probs = external_rdm_probs

        text_part_mask = text_part_mask_generation(
            input_ids, special_token_ids, attention_mask
        )

        if mlm_prob is not None:
            # Tokens where random_probs < mlm_prob and in text part are selected
            selected_for_masking = (random_probs < mlm_prob).to(
                torch.long
            ) * text_part_mask
            if extra_masked_flags is not None:
                # Add extra tokens to mask, ensuring they are in text part
                selected_for_masking = (
                    (selected_for_masking + extra_masked_flags * text_part_mask) > 0
                ).to(torch.long)
            if exclude_flags is not None:
                # Exclude specified tokens from masking
                selected_for_masking = selected_for_masking * (1 - exclude_flags)
        else:
            # If mlm_prob is None, use extra_masked_flags as the base selection
            assert extra_masked_flags is not None
            selected_for_masking = extra_masked_flags * text_part_mask
            if exclude_flags is not None:
                selected_for_masking = selected_for_masking * (1 - exclude_flags)

        selected_for_masking_bool = selected_for_masking.to(torch.bool)

        mlm_labels = input_ids.clone()
        mlm_labels.masked_fill_(~selected_for_masking_bool, -100)

        # Prepare masked input IDs
        masked_input_ids = input_ids.clone()
        mlm_mask_prob = 1.0 - mlm_rdm_prob - mlm_keep_prob  # Probability for [MASK]
        assert mlm_mask_prob >= 0.0  # Ensure probabilities sum to <= 1

        if resample_nonmask:
            # Use new random numbers to determine replacement type
            split_probs = torch.rand(input_ids.shape, device=input_ids.device)
            if mlm_mask_prob > 1e-5:
                # Replace with [MASK] where split_probs < mlm_mask_prob
                replace_with_mask = (
                    split_probs < mlm_mask_prob
                ) & selected_for_masking_bool
                masked_input_ids.masked_fill_(replace_with_mask, mask_token_id)
            if mlm_rdm_prob > 1e-5:
                # Replace with random token in the next probability interval
                replace_with_random = (
                    (split_probs >= mlm_mask_prob)
                    & (split_probs < mlm_mask_prob + mlm_rdm_prob)
                    & selected_for_masking_bool
                )
                random_tokens = torch.randint(
                    0, vocab_size, input_ids.shape, device=input_ids.device
                )
                masked_input_ids = torch.where(
                    replace_with_random, random_tokens, masked_input_ids
                )
            # Remaining selected tokens (split_probs >= mlm_mask_prob + mlm_rdm_prob) stay unchanged
        else:
            # Use original random_probs to partition [0, mlm_prob) into replacement intervals
            if mlm_mask_prob > 1e-5:
                threshold_mask = mlm_prob * mlm_mask_prob
                replace_with_mask = (
                    random_probs < threshold_mask
                ) & selected_for_masking_bool
                masked_input_ids.masked_fill_(replace_with_mask, mask_token_id)
            if mlm_rdm_prob > 1e-5:
                threshold_rdm_lower = mlm_prob * mlm_mask_prob
                threshold_rdm_upper = mlm_prob * (mlm_mask_prob + mlm_rdm_prob)
                replace_with_random = (
                    (random_probs >= threshold_rdm_lower)
                    & (random_probs < threshold_rdm_upper)
                    & selected_for_masking_bool
                )
                random_tokens = torch.randint(
                    0, vocab_size, input_ids.shape, device=input_ids.device
                )
                masked_input_ids = torch.where(
                    replace_with_random, random_tokens, masked_input_ids
                )
            # Tokens with random_probs >= threshold_rdm_upper but < mlm_prob stay unchanged

        return masked_input_ids, selected_for_masking, mlm_labels, random_probs


def build_param_groups(
    model, enc_lr, dec_lr, weight_decay, norm_keywords=("norm", "layernorm", "rmsnorm")
):
    encoder_decay, encoder_no_decay = [], []
    decoder_decay, decoder_no_decay = [], []

    ENCODER_PREFIXES = ("model.", "head.")
    DECODER_PREFIXES = ("dec_head.", "decoder.", "decoder_heads.")

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # ----- decide whether this param gets weight-decay -----
        is_bias = name.endswith(".bias")
        is_norm = any(kw in name.lower() for kw in norm_keywords)
        no_decay = is_bias or is_norm

        # ----- route to encoder vs decoder bucket -----
        if name.startswith(ENCODER_PREFIXES):
            (encoder_no_decay if no_decay else encoder_decay).append(param)
        elif name.startswith(DECODER_PREFIXES):
            (decoder_no_decay if no_decay else decoder_decay).append(param)
        else:
            # anything that slips through (e.g. embeddings) → encoder side
            (encoder_no_decay if no_decay else encoder_decay).append(param)

    return [
        {"params": encoder_decay, "lr": enc_lr, "weight_decay": weight_decay},
        {"params": encoder_no_decay, "lr": enc_lr, "weight_decay": 0.0},
        {"params": decoder_decay, "lr": dec_lr, "weight_decay": weight_decay},
        {"params": decoder_no_decay, "lr": dec_lr, "weight_decay": 0.0},
    ]


def save_checkpoint_val(
    step: int,
    score: float,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
) -> str:
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "ndcg@10": score,
    }
    filepath = os.path.join(
        checkpoint_path, f"checkpoint_step_{step}_ndcg_{score:.4f}.pt"
    )
    torch.save(checkpoint, filepath)
    return filepath


def update_checkpoint_tracking_val(
    step: int,
    score: float,
    checkpoint_scores: list,
    max_checkpoints: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
) -> list:
    # Create a new list to maintain purity
    updated_checkpoint_scores = checkpoint_scores.copy()

    if len(updated_checkpoint_scores) < max_checkpoints:
        filepath = save_checkpoint(step, score, model, optimizer, checkpoint_path)
        heappush(updated_checkpoint_scores, (score, step, filepath))
    elif score > updated_checkpoint_scores[0][0]:  # Compare with lowest score
        # Remove lowest scoring checkpoint
        _, old_step, old_filepath = heappop(updated_checkpoint_scores)
        if os.path.exists(old_filepath):
            os.remove(old_filepath)
        # Save new checkpoint
        filepath = save_checkpoint(step, score, model, optimizer, checkpoint_path)
        heappush(updated_checkpoint_scores, (score, step, filepath))

    return updated_checkpoint_scores
