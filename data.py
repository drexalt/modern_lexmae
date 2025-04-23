import torch
from transformers import AutoTokenizer


class LexMAECollate:
    def __init__(self, tokenizer, max_length=512, text_key="text", vocab_size=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_key = text_key
        self.vocab_size = vocab_size if vocab_size is not None else len(tokenizer)

    def __call__(self, batch):
        texts = [ex[self.text_key] for ex in batch]
        encodings = self.tokenizer(
            texts,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encodings["input_ids"]  # [batch_size, max_length]
        attention_mask = encodings["attention_mask"]  # [batch_size, max_length]
        batch_size = input_ids.size(0)
        device = input_ids.device

        bag_word_weights = []
        for b in range(batch_size):
            seq_len = attention_mask[b].sum().item()

            if seq_len > 2:
                content_tokens = input_ids[b, 1 : seq_len - 1]
                total_tokens = seq_len - 2

                unique_tokens = torch.unique(content_tokens)

                weight = torch.zeros(self.vocab_size, device=device)
                weight[unique_tokens] = 1 / total_tokens
            else:
                weight = torch.zeros(self.vocab_size, device=device)

            bag_word_weights.append(weight)

        bag_word_weight = torch.stack(
            bag_word_weights, dim=0
        )  # [batch_size, vocab_size]

        encodings["bag_word_weight"] = bag_word_weight

        return encodings
