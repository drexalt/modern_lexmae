import torch
from transformers import AutoTokenizer


class LexMAECollate:
    def __init__(self, tokenizer, max_length=512, text_key="text"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_key = text_key

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

        return encodings
