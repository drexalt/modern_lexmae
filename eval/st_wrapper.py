import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


# Triton kernel
@triton.jit
def sparse_activation_kernel(
    logits_ptr,
    mask_ptr,
    output_ptr,
    indices_ptr,
    batch_size,
    seq_len,
    vocab_size,
    logit_batch_stride,
    logit_seq_stride,
    logit_vocab_stride,
    mask_batch_stride,
    mask_seq_stride,
    output_batch_stride,
    output_vocab_stride,
    indices_batch_stride,
    indices_vocab_stride,
    BLOCK_V: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    compute_dtype = (
        tl.float32 if output_ptr.dtype.element_ty == tl.float32 else tl.bfloat16
    )
    pid_batch = tl.program_id(0)
    pid_v_chunk = tl.program_id(1)

    v_offset = pid_v_chunk * BLOCK_V
    v_indices = v_offset + tl.arange(0, BLOCK_V)
    batch_idx = pid_batch
    v_mask = v_indices < vocab_size
    v_safe = tl.where(v_mask, v_indices, vocab_size - 1)

    max_accumulator = tl.full((BLOCK_V,), -float("inf"), dtype=compute_dtype)
    argmax_accumulator = tl.full((BLOCK_V,), -1, dtype=tl.int32)

    for s_offset in range(0, seq_len, BLOCK_S):
        s_indices = s_offset + tl.arange(0, BLOCK_S)
        s_mask = s_indices < seq_len
        s_safe = tl.where(s_mask, s_indices, seq_len - 1)
        mask_offsets = batch_idx * mask_batch_stride + s_indices * mask_seq_stride
        mask = tl.load(mask_ptr + mask_offsets, mask=s_indices < seq_len, other=0.0).to(
            compute_dtype
        )

        logit_offsets = (
            (batch_idx * logit_batch_stride)
            + (s_safe[:, None] * logit_seq_stride)
            + (v_safe[None, :] * logit_vocab_stride)
        )
        logits = tl.load(
            logits_ptr + logit_offsets,
            mask=s_mask[:, None] & v_mask[None, :],
            other=-float("inf"),
        ).to(compute_dtype)

        activated = tl.math.log(1 + tl.maximum(logits, 0.0)) * mask[:, None]
        chunk_max = tl.max(activated, axis=0).to(compute_dtype)
        chunk_argmax = tl.argmax(activated, axis=0)
        update_mask = chunk_max > max_accumulator
        max_accumulator = tl.where(update_mask, chunk_max, max_accumulator)
        argmax_accumulator = tl.where(
            update_mask, s_offset + chunk_argmax, argmax_accumulator
        )

    output_offsets = batch_idx * output_batch_stride + v_safe * output_vocab_stride
    indices_offsets = batch_idx * indices_batch_stride + v_safe * indices_vocab_stride
    tl.store(output_ptr + output_offsets, max_accumulator, mask=v_mask)
    tl.store(indices_ptr + indices_offsets, argmax_accumulator, mask=v_mask)


# Define the forward pass as a Triton operation
@triton_op("custom::sparse_activation", mutates_args={})
def sparse_activation(
    logits: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    batch_size, seq_len, vocab_size = logits.shape
    device = logits.device

    values = torch.empty(batch_size, vocab_size, device=device, dtype=logits.dtype)
    argmax_indices = torch.empty(
        batch_size, vocab_size, device=device, dtype=torch.int32
    )

    grid = lambda meta: (batch_size, triton.cdiv(vocab_size, meta["BLOCK_V"]))
    wrap_triton(sparse_activation_kernel)[grid](
        logits,
        attention_mask.to(logits.dtype),
        values,
        argmax_indices,
        batch_size,
        seq_len,
        vocab_size,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        attention_mask.stride(0),
        attention_mask.stride(1),
        values.stride(0),
        values.stride(1),
        argmax_indices.stride(0),
        argmax_indices.stride(1),
        BLOCK_V=1024,
        BLOCK_S=128,
    )

    sparse_activation._argmax_indices = argmax_indices  # Temporary storage
    return values


class ST_LexMAEModule(nn.Module):
    def __init__(self, model, tokenizer, max_length=256):
        """
        Initialize the LexMAE module for sparse embedding computation.

        Args:
            model: Pre-trained ModernBertForLexMAE instance.
            tokenizer: Associated tokenizer (e.g., AutoTokenizer).
            max_length: Maximum sequence length for tokenization (default: 256).
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    def forward(self, features: dict) -> dict:
        """
        Compute sparse embeddings from encoder logits for SentenceTransformer.

        Args:
            features: Dictionary with 'input_ids' and 'attention_mask' (torch.Tensor).

        Returns:
            features: Updated dictionary with 'sentence_embedding' key.
        """
        model_inputs = {
            "input_ids": features["input_ids"],
            "attention_mask": features["attention_mask"],
        }
        device = next(self.model.parameters()).device
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        attention_mask = model_inputs["attention_mask"].unsqueeze(-1)

        self.model.eval()
        with torch.inference_mode():
            outputs = self.model(**model_inputs, disable_decoding=True)
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]

            logits.relu_()
            logits.log1p_()

            logits = logits * attention_mask
            # Max-pool over sequence dimension to get [batch_size, vocab_size]
            values = torch.amax(logits, dim=1)

            top_values, _ = torch.topk(values, k=256, dim=-1)
            threshold = top_values[..., -1, None]
            values = values * (values >= threshold)

            features["sentence_embedding"] = values
        return features

    # def forward(self, features: dict) -> dict:
    #     """
    #     Compute sparse embeddings from encoder logits for SentenceTransformer.

    #     Args:
    #         features: Dictionary with 'input_ids' and 'attention_mask' (torch.Tensor).

    #     Returns:
    #         features: Updated dictionary with 'sentence_embedding' key.
    #     """
    #     model_inputs = {
    #         "input_ids": features["input_ids"],
    #         "attention_mask": features["attention_mask"],
    #     }
    #     device = next(self.model.parameters()).device
    #     model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    #     attention_mask = model_inputs["attention_mask"].unsqueeze(-1)
    #     self.model.eval()
    #     with torch.inference_mode():
    #         with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
    #             outputs = self.model(**model_inputs, disable_decoding=True)
    #         sparse_embedding = sparse_activation(
    #             outputs.logits, model_inputs["attention_mask"]
    #         )

    #     features["sentence_embedding"] = sparse_embedding.cpu()
    #     return features

    def tokenize(self, texts):
        """
        Tokenize input texts for SentenceTransformer.

        Args:
            texts: List of strings to tokenize.

        Returns:
            Dictionary with tokenized tensors ('input_ids', 'attention_mask').
        """
        return self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def encode(self, texts, batch_size=32, **kwargs):
        """
        Encode texts into sparse embeddings.

        Args:
            texts: List of strings to encode.
            batch_size: Batch size for processing (default: 32).

        Returns:
            Sparse embeddings as NumPy array [len(texts), vocab_size].
        """
        inputs = self.tokenize(texts).to(self.model.device)
        with torch.inference_mode():
            outputs = self.model(**inputs, disable_decoding=True)
            logits = outputs.logits

            relu_logits = torch.relu(logits)
            transformed_logits = torch.log1p(relu_logits)
            attention_mask = inputs["attention_mask"].unsqueeze(-1).bool()
            masked_transformed_logits = torch.where(
                attention_mask,
                transformed_logits,
                torch.tensor(float("-inf"), device=logits.device),
            )
            sparse_embedding, _ = torch.max(masked_transformed_logits, dim=1)
            return sparse_embedding.cpu().numpy()
