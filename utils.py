import torch
from functools import reduce


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
