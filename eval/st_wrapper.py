# pyright: basic
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


class ST_LexMAEModule(nn.Module):
    def __init__(self, model, tokenizer, max_length=256, top_k=2048):
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
        self.top_k = top_k

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
            outputs = self.model(**model_inputs)
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]

            logits.relu_()
            logits.log1p_()

            logits = logits * attention_mask
            # Max-pool over sequence dimension to get [batch_size, vocab_size]
            values = torch.amax(logits, dim=1)

            top_values, _ = torch.topk(values, k=self.top_k, dim=-1)
            threshold = top_values[..., -1, None]
            values = values * (values >= threshold)

            features["sentence_embedding"] = values.cpu()
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
