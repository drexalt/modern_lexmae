from sentence_transformers import SentenceTransformer
from .st_wrapper import ST_LexMAEModule
import torch


def validate_lexmae(evaluator, model, tokenizer, device):
    """
    Run NanoBEIR evaluation on the LexMAE model for zero-shot retrieval.

    Args:
        evaluator: NanoBEIR evaluator instance.
        model: Pre-trained ModernBertForLexMAE instance.
        tokenizer: Associated tokenizer.
        device: Torch device (e.g., 'cuda' or 'cpu').

    Returns:
        Dictionary of evaluation metrics (e.g., NDCG@10, MRR@10).
    """
    # Create the SentenceTransformer module
    st_module = ST_LexMAEModule(model, tokenizer, max_length=tokenizer.model_max_length)
    st_model = SentenceTransformer(modules=[st_module]).to(device)

    # Evaluate with no gradient computation
    with torch.inference_mode():
        results = evaluator(st_model)

    # Clean up
    del st_module, st_model
    torch.cuda.empty_cache()

    # Extract metrics
    primary_metrics = {
        "ndcg@10": results["NanoBEIR_mean_dot_ndcg@10"],
        "mrr@10": results["NanoBEIR_mean_dot_mrr@10"],
        "map@100": results["NanoBEIR_mean_dot_map@100"],
    }
    supplementary_metrics = {
        "recall@10": results["NanoBEIR_mean_dot_recall@10"],
        "precision@1": results["NanoBEIR_mean_dot_precision@1"],
    }
    return {**primary_metrics, **supplementary_metrics}
