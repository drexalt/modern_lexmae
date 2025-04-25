import os
from datetime import datetime
from bert_reference import BertForEDMLM
from utils import mlm_input_ids_masking_onthefly, update_checkpoint_tracking
from peach.enc_utils.enc_learners import LearnerMixin
from datasets import load_dataset
from data import LexMAECollateDupMAE
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig
import hydra
from omegaconf import DictConfig
import heavyball
from heavyball.utils import trust_region_clip_, rmsnorm_clip_
import torch
import wandb
from tqdm import tqdm
from sentence_transformers.evaluation import NanoBEIREvaluator
from sentence_transformers.similarity_functions import dot_score
from eval.validate import validate_lexmae


class LexmaeLearner(LearnerMixin):
    def __init__(self, cfg, config, tokenizer, encoder, query_encoder=None):
        super(LexmaeLearner, self).__init__(
            cfg,
            config,
            tokenizer,
            encoder,
            query_encoder,
        )
        self.cfg = cfg
        self.mask_token_id = self.tokenizer.mask_token_id
        self.special_token_ids = [
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
        ]
        self.vocab_size = len(self.tokenizer)
        self.base_model_prefix = "encoder." + encoder.base_model_prefix

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        bag_word_weight=None,
        masked_input_ids=None,
        masked_flags=None,
        mlm_labels=None,
        dec_masked_input_ids=None,
        dec_attention_mask=None,
        dec_masked_flags=None,
        dec_mlm_labels=None,
        training_progress=None,
        training_mode=None,
        **kwargs,
    ):
        if training_mode is None:
            return self.encoder(
                input_ids,
                attention_mask,
                token_type_ids,
                position_ids,
            )

        # embedding
        dict_for_meta, dict_for_loss = {}, {}

        dict_for_meta["mlm_enc_loss_weight"] = self.model_args.mlm_enc_loss_weight
        dict_for_meta["mlm_dec_loss_weight"] = self.model_args.mlm_dec_loss_weight

        # encoder masking and forward
        if masked_input_ids is None:  # on-the-fly generation
            masked_input_ids, masked_flags, mlm_labels, rdm_probs = (
                mlm_input_ids_masking_onthefly(
                    input_ids,
                    attention_mask,
                    self.mask_token_id,
                    self.special_token_ids,
                    mlm_prob=self.model_args.enc_mlm_prob,
                    mlm_rdm_prob=0.1,
                    mlm_keep_prob=0.1,
                    vocab_size=self.vocab_size,
                    external_rdm_probs=None,
                    extra_masked_flags=None,
                    resample_nonmask=False,
                )
            )
            enc_mlm_prob = self.model_args.enc_mlm_prob
        else:
            enc_mlm_prob = self.model_args.data_mlm_prob

        enc_outputs = self.encoder(
            masked_input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            enc_mlm_labels=mlm_labels,
            disable_encoding=False,
            disable_decoding=True,
            bag_word_weight=bag_word_weight,
        )
        dict_for_loss["mlm_enc_loss"] = enc_outputs["loss"]
        dict_for_loss["bow_loss"] = enc_outputs["bow_loss"]
        # decoder masking and forward
        if dec_masked_input_ids is None:
            if self.model_args.dec_mlm_overlap == "random":
                extra_masked_flags, exclude_flags = None, None
                dec_mlm_prob = self.model_args.dec_mlm_prob
            elif self.model_args.dec_mlm_overlap == "inclusive":
                assert self.model_args.dec_mlm_prob >= enc_mlm_prob
                extra_masked_flags, exclude_flags = masked_flags, None
                dec_mlm_prob = (self.model_args.dec_mlm_prob - enc_mlm_prob) / (
                    1.0 - enc_mlm_prob
                )
            elif self.model_args.dec_mlm_overlap == "exclusive":
                assert self.model_args.dec_mlm_prob <= (1.0 - enc_mlm_prob)
                extra_masked_flags, exclude_flags = None, masked_flags
                dec_mlm_prob = self.model_args.dec_mlm_prob / (1.0 - enc_mlm_prob)
            else:
                raise NotImplementedError

            dec_masked_input_ids, dec_masked_flags, dec_mlm_labels, _ = (
                mlm_input_ids_masking_onthefly(
                    input_ids,
                    attention_mask,
                    self.mask_token_id,
                    self.special_token_ids,
                    mlm_prob=dec_mlm_prob,
                    mlm_rdm_prob=0.1,
                    mlm_keep_prob=0.1,
                    vocab_size=self.vocab_size,
                    external_rdm_probs=None,
                    extra_masked_flags=extra_masked_flags,
                    exclude_flags=exclude_flags,
                    resample_nonmask=True,
                )
            )

        dec_attention_mask = (
            dec_attention_mask if dec_attention_mask is not None else attention_mask
        )
        dec_outputs = self.encoder(
            dec_input_ids=dec_masked_input_ids,
            dec_attention_mask=dec_attention_mask,
            dec_token_type_ids=None,
            dec_position_ids=None,
            enc_cls_rep=enc_outputs["sentence_embedding"],
            enc_hidden_states=enc_outputs["hidden_states"],
            dec_mlm_labels=dec_mlm_labels,
            disable_encoding=True,
            disable_decoding=False,
        )
        dict_for_loss["mlm_dec_loss"] = dec_outputs["dec_loss"]
        dict_for_loss["loss"] = (
            self.cfg.mlm_enc_loss_weight * dict_for_loss["mlm_enc_loss"]
            + self.cfg.mlm_dec_loss_weight * dict_for_loss["mlm_dec_loss"]
            + self.cfg.bow_loss_weight * dict_for_loss["bow_loss"]
        )

        dict_for_meta.update(dict_for_loss)
        return dict_for_meta


def train(cfg, train_dataloader, model, optimizer, device, tokenizer):
    model.train()
    model.zero_grad()
    if cfg.wandb:
        wandb.init(
            project=cfg.wandb_project,
            config={
                "batch_size": cfg.batch_size,
                "learning_rate": cfg.optimizer.learning_rate,
                "warmup_steps": cfg.optimizer.warmup_steps,
                "head_layers": cfg.n_head_layers,
                "optimizer": optimizer.__class__.__name__,
                "accumulation_steps": cfg.accumulation_steps,
                "enc_mlm_prob": cfg.enc_mlm_prob,
                "dec_mlm_prob": cfg.dec_mlm_prob,
                "mlm_enc_loss_weight": cfg.mlm_enc_loss_weight,
                "mlm_dec_loss_weight": cfg.mlm_dec_loss_weight,
                "bow_loss_weight": cfg.bow_loss_weight,
            },
        )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_directory = os.path.join(
        hydra.utils.to_absolute_path(cfg.checkpoint.checkpoint_path), timestamp
    )
    os.makedirs(checkpoint_directory, exist_ok=True)
    checkpoint_losses = []
    evaluator = NanoBEIREvaluator(
        dataset_names=cfg.evaluation.datasets,
        score_functions={"dot": dot_score},
        batch_size=cfg.evaluation.batch_size,
        show_progress_bar=True,
    )

    for epoch in range(cfg.num_train_epochs):
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}  # Move batch to device
            loss_dict = model(training_mode="pre-training", **batch)
            loss = loss_dict["loss"]

            loss = loss / cfg.accumulation_steps
            loss.backward()

            if (step + 1) % cfg.accumulation_steps == 0 or (step + 1) == len(
                train_dataloader
            ):
                optimizer.step()
                optimizer.zero_grad()

            if cfg.wandb and step % cfg.log_every == 0:
                wandb.log(
                    {
                        "loss": loss_dict["loss"].item(),
                        "mlm_enc_loss": loss_dict["mlm_enc_loss"].item(),
                        "mlm_dec_loss": loss_dict["mlm_dec_loss"].item(),
                        "bow_loss": loss_dict["bow_loss"].item(),
                    },
                    step=(epoch * len(train_dataloader)) + step,
                )

            if (step + 1) % cfg.evaluation.eval_every_steps == 0 or step == 5:
                model.eval()
                val_results = validate_lexmae(
                    evaluator,
                    model,
                    tokenizer,
                    device,
                )
                model.train()

                if cfg.wandb:
                    # Flatten results for wandb logging
                    wandb.log(
                        {
                            "validation/ndcg@10": val_results["ndcg@10"],
                            "validation/mrr@10": val_results["mrr@10"],
                            "validation/map@100": val_results["map@100"],
                            **{
                                f"validation/supplementary/{k}": v
                                for k, v in val_results.items()
                                if k not in ["ndcg@10", "mrr@10", "map@100"]
                            },
                        },
                        step=(epoch * len(train_dataloader)) + step,
                    )

            if (
                cfg.checkpoint.checkpoint_every > 0
                and (step + 1) % cfg.checkpoint.checkpoint_every == 0
            ):
                checkpoint_losses = update_checkpoint_tracking(
                    step=(epoch * len(train_dataloader)) + step,
                    loss=loss_dict["loss"].item(),
                    checkpoint_losses=checkpoint_losses,
                    max_checkpoints=cfg.checkpoint.max_to_keep,
                    model=model,
                    optimizer=optimizer,
                    checkpoint_path=checkpoint_directory,
                )


@hydra.main(config_path="conf", config_name="bert")
def main(cfg: DictConfig):
    config = AutoConfig.from_pretrained(cfg.model.model_name_or_path)
    config.n_head_layers = cfg.n_head_layers
    encoder = BertForEDMLM.from_pretrained(cfg.model.model_name_or_path, config=config)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path)
    model = LexmaeLearner(cfg, config, tokenizer, encoder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    encoder_params = []
    decoder_params = []
    base_model_prefix = model.encoder.base_model_prefix
    encoder_head_prefix = "head."
    for name, param in model.encoder.named_parameters():
        if param.requires_grad:
            if name.startswith(base_model_prefix + ".") or name.startswith(
                encoder_head_prefix
            ):
                encoder_params.append(param)
            else:
                decoder_params.append(param)

    optimizer_grouped_parameters = [
        {"params": encoder_params, "lr": cfg.optimizer.enc_learning_rate},
        {"params": decoder_params, "lr": cfg.optimizer.dec_learning_rate},
    ]

    dataset = load_dataset("BeIR/msmarco", "corpus", split="corpus")
    train_dataloader = DataLoader(
        dataset,
        num_workers=4,
        batch_size=cfg.batch_size,
        collate_fn=LexMAECollateDupMAE(tokenizer, max_length=cfg.model.max_length),
        pin_memory=True,
    )
    heavyball.utils.compile_mode = None
    heavyball.utils.set_torch()

    # optimizer = torch.optim.AdamW(
    #     model.encoder.parameters(),
    #     lr=cfg.optimizer.learning_rate,
    #     weight_decay=cfg.optimizer.weight_decay,
    # )
    optimizer = heavyball.ForeachSFAdamW(
        optimizer_grouped_parameters,
        lr=cfg.optimizer.learning_rate,
        warmup_steps=cfg.optimizer.warmup_steps,
        weight_decay=cfg.optimizer.weight_decay,
        foreach=True,
    )
    train(cfg, train_dataloader, model, optimizer, device, tokenizer)


if __name__ == "__main__":
    main()
