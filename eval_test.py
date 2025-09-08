# pyright: basic
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import NanoBEIREvaluator
from sentence_transformers.similarity_functions import dot_score
from eval.st_wrapper import ST_LexMAEModule
import torch
from eval.validate import validate_lexmae
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    BertModel,
    AutoConfig,
    AutoModel,
)
from presplade.bert_lexmae import BertAdapter
from presplade.neobert_lexmae import NeoBertAdapter
from general_train import LexmaeLearner
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="neobert")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = "chandar-lab/NeoBERT"
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.n_head_layers = cfg.n_head_layers
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    encoder = NeoBertAdapter.from_pretrained(
        model_name, config=config, trust_remote_code=True
    )

    checkpoint = torch.load(
        "/root/code/python/stage/modern_lexmae/checkpoints/neobert-lexmae/20250904_070620/checkpoint_step_209999_loss_0.1935.pt",
        weights_only=False,
    )

    state_dict = checkpoint.get("model", checkpoint)

    model = LexmaeLearner(cfg, config, tokenizer, encoder)
    model.load_state_dict(state_dict)
    model.eval()

    evaluator = NanoBEIREvaluator(
        dataset_names=["msmarco", "scifact"],
        score_functions={"dot": dot_score},
        batch_size=8,
        show_progress_bar=True,
    )

    results = validate_lexmae(
        evaluator, model, tokenizer, device, max_length=256, top_k=1024
    )

    print(results)


if __name__ == "__main__":
    main()
