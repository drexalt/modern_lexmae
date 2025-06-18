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
    st_model = SentenceTransformer(modules=[st_module], device=device).eval()

    # Evaluate with no gradient computation
    with torch.inference_mode():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
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
        "msmarco_mrr@10": results["NanoMSMARCO_dot_mrr@10"],
        "msmarco_ndcg@10": results["NanoMSMARCO_dot_ndcg@10"],
        "msmarco_map@100": results["NanoMSMARCO_dot_map@100"],
    }
    return {**primary_metrics, **supplementary_metrics}
