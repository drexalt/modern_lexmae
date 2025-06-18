import torch
from functools import reduce
import os
from heapq import heappush, heapreplace


def text_part_mask_generation(input_ids, special_token_ids, attention_mask):
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


def masked_pool(tensor, mask, high_rank=True, method="max"):
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
            mask_text_part = text_part_mask_generation(
                input_ids, special_token_ids, attention_mask
            )
        pooled_enc_logits = masked_pool(
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


## DUP-MAE
def ot_embedding(logits: torch.Tensor, attention_mask: torch.Tensor):
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
def bow_ot_loss(ot_embedding: torch.Tensor, bag_word_weight: torch.Tensor):
    """
    Cross‑entropy between pooled logits and target BoW distribution.
    Args:
        ot_embedding    – [bs, vocab]
        bag_word_weight – [bs, vocab]   (row‑normalised to 1.0)
    """
    log_probs = torch.log_softmax(ot_embedding, dim=-1)
    return torch.mean(-torch.sum(bag_word_weight * log_probs, dim=-1))


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
