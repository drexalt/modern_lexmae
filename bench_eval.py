# pyright: basic
import torch
import hydra
from omegaconf import DictConfig
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sentence_transformers.evaluation import NanoBEIREvaluator
from sentence_transformers.similarity_functions import dot_score
from eval.validate import validate_lexmae


@hydra.main(config_path="conf", config_name="bert")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prefer the Transformers checkpoint (has MLM logits)
    model_id = "Shitao/RetroMAE_MSMARCO"
    # If you insist on the SentenceTransformers port, use its transformer subfolder:
    # model_id = "nthakur/RetroMAE_MSMARCO_finetune/0_Transformer"

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = (
        AutoModelForMaskedLM.from_pretrained(
            model_id, trust_remote_code=True, attn_implementation="eager"
        )
        .to(device)
        .eval()
    )

    evaluator = NanoBEIREvaluator(
        dataset_names=["msmarco", "scifact"],
        score_functions={"dot": dot_score},
        batch_size=32,
        show_progress_bar=True,
    )

    results = validate_lexmae(
        evaluator, model, tokenizer, device, max_length=256, top_k=1024
    )
    print(results)


if __name__ == "__main__":
    main()
