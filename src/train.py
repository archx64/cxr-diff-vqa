import argparse
import random
from pathlib import Path
import logging
import json
import yaml
import os
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

# Neptune AI Imports
try:
    import neptune
    from neptune.types import File
except ImportError:
    print("Neptune not installed. Logging disabled.")

# Try to import the notebook-friendly version of tqdm
from tqdm import tqdm

# Import project files
from lib.dataset import DiffVQADataset
from lib.utils import make_loader, setup_logging, run_nlg_evaluation, NEPTUNE_API_TOKEN, NEPTUNE_PROJECT
from lib.model.vqa import DiffVQAModel
from lib.losses import heatmap_kl, info_nce_token_sets
from lib.negate import negate_question

# Global logger
logger = logging.getLogger()

# Global Neptune run variable
run = None


def parse_args():
    """
    Parses command-line arguments and loads configuration from a YAML file.
    """
    parser = argparse.ArgumentParser(description="Train a DRIFT-VQA model.")
    parser.add_argument("--config", type=str, help="Path to YAML config")

    # --- Split-Specific Files ---
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--train_pairs_csv", type=str)
    parser.add_argument("--train_meta_csv", type=str)
    parser.add_argument("--val_pairs_csv", type=str)
    parser.add_argument("--val_meta_csv", type=str)
    parser.add_argument("--test_pairs_csv", type=str)
    parser.add_argument("--test_meta_csv", type=str)
    parser.add_argument("--keyinfo_json", type=str)
    parser.add_argument(
        "--ckpt", type=str, default="", help="Path to pretrained backbone weights"
    )

    # --- Model Params ---
    parser.add_argument("--backbone", type=str)
    parser.add_argument("--text_encoder", type=str)
    parser.add_argument("--text_model_name", type=str)
    parser.add_argument("--text_finetune", action="store_true")
    parser.add_argument(
        "--text_dim",
        type=int,
    )
    parser.add_argument(
        "--text_proj_dim",
        type=int,
    )
    parser.add_argument("--max_ans_len", type=int)
    parser.add_argument("--topk", type=int)

    # --- Ablation Flags ---
    parser.add_argument(
        "--ablation_no_direction", action="store_true", help="Use only R_abs features"
    )

    # --- Training Params ---
    parser.add_argument(
        "--bs",
        type=int,
    )
    parser.add_argument(
        "--lr",
        type=float,
    )
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--epochs_mrm", type=int)
    parser.add_argument("--epochs_warmup", type=int)
    parser.add_argument(
        "--epochs_vqa",
        type=int,
    )
    parser.add_argument("--mask_ratio", type=float)

    # --- Loss Weights ---
    parser.add_argument(
        "--main_loss_weight",
        type=float,
    )
    parser.add_argument(
        "--lambda_mrm",
        type=float,
    )
    parser.add_argument("--lambda_align", type=float)
    parser.add_argument("--lambda_cf", type=float)

    parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument(
    #     "--neptune_project",
    #     type=str,
    #     help="Neptune project name",
    # )

    args = parser.parse_args()

    # YAML overrides CLI defaults
    if args.config and Path(args.config).exists():
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
        for k, v in cfg.items():
            setattr(args, k, v)

    # Sanity check
    needed = [
        "data_root",
        "train_pairs_csv",
        "train_meta_csv",
        "val_pairs_csv",
        "val_meta_csv",
    ]
    missing = [k for k in needed if not getattr(args, k, None)]
    if missing:
        raise SystemExit(f"Missing required settings: {', '.join(missing)}")

    return args


def seed_all(s=42):
    random.seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


def tokenize_questions(text_model, batch_questions, device=None):
    """
    Helper to tokenize a batch of questions using the HuggingFace tokenizer.
    """
    enc = text_model.tokenize(batch_questions)
    if device is not None:
        enc = {k: v.to(device) for k, v in enc.items()}
    return enc


def run_epoch(
    stage,
    model,
    loader,
    optimizer,
    scaler,
    device,
    main_loss_weight=1.0,
    lambda_mrm=0.1,
    lambda_align=0.05,
    lambda_cf=0.05,
    lambda_gate=1e-3,
):
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc=f"Stage {stage}", leave=False, file=sys.stdout)

    for i, batch in enumerate(pbar):
        img_cur = batch["img_cur"].to(device)
        img_ref = batch["img_ref"].to(device)
        qs = batch["question"]
        y_seq = batch["answer_ids"].to(device)

        # 1. Tokenize questions
        tokens = tokenize_questions(model.text, qs, device=device)

        # 2. Optimization: Only prepare CF inputs if we are actually using the loss
        # This saves significant compute if lambda_cf is 0.0
        if lambda_cf > 0:
            qs_cf = [negate_question(q) for q in qs]
            tokens_cf = tokenize_questions(model.text, qs_cf, device=device)
            img_cur_cf, img_ref_cf = img_ref, img_cur  # Swap images
        else:
            tokens_cf = None
            img_cur_cf, img_ref_cf = None, None

        with torch.autocast(
            device_type=device.type if device.type == "cuda" else "cpu",
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
            enabled=(device.type == "cuda"),
        ):
            # Forward Pass (Main)
            out = model(img_ref, img_cur, tokens)

            # Calculate Auxiliary Losses
            loss_mrm = out["loss_mrm"]
            loss_align = model.drs.alignment_loss(
                out["r_pos"], out["r_neg"], out["r_abs"], out["signed"]
            )
            loss_gate = out["gate_l1"]

            # Calculate Main Decoder Loss
            _, main_loss = model.head(out["sel_tokens"], targets=y_seq)

            # Conditional Counterfactual Loss Calculation
            loss_cf_combined = 0.0
            if lambda_cf > 0:
                # Only run the second forward pass if necessary
                out_cf = model(img_ref_cf, img_cur_cf, tokens_cf)
                loss_hkl = heatmap_kl(out["heatmap"], out_cf["heatmap"])
                loss_nce = info_nce_token_sets(out["patches"], out_cf["patches"])
                loss_cf_combined = loss_hkl + loss_nce

            # Combine Losses
            if stage == "mrm":
                loss = loss_mrm
            elif stage == "warmup":
                loss = (
                    (main_loss_weight * main_loss)
                    + lambda_mrm * loss_mrm
                    + lambda_align * loss_align
                    + lambda_gate * loss_gate
                )
            else:  # stage == "vqa"
                loss = (
                    (main_loss_weight * main_loss)
                    + lambda_mrm * loss_mrm
                    + lambda_align * loss_align
                    + lambda_cf * loss_cf_combined  # Will be 0 if disabled
                    + lambda_gate * loss_gate
                )

        # Backward Pass
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        # Neptune Logging (if enabled)
        if run and (i % 10 == 0 or i == 0):
            run[f"train/{stage}/loss"].append(loss.item())
            run[f"train/{stage}/lr"].append(optimizer.param_groups[0]["lr"])

        avg_loss = running_loss / (i + 1)
        pbar.set_description(
            f"Stage {stage} | Batch Loss: {loss.item():.4f} | Avg: {avg_loss:.4f}"
        )

    return running_loss / len(loader)


def evaluate(model, loader, device, vocab):
    """Runs evaluation on the validation set."""
    model.eval()
    gts = {"info": {}, "images": [], "annotations": []}
    res = []
    sample_id_counter = 0

    pbar = tqdm(loader, desc="Validation", leave=False, file=sys.stdout)
    with torch.no_grad():
        for batch in pbar:
            img_cur = batch["img_cur"].to(device)
            img_ref = batch["img_ref"].to(device)
            qs = batch["question"]
            ground_truth_ids = batch["answer_ids"].cpu()

            tokens = tokenize_questions(model.text, qs, device=device)
            out = model(img_ref, img_cur, tokens)

            _, preds_ids = model.head(out["sel_tokens"])
            preds_ids = preds_ids.cpu().tolist()

            for i in range(len(qs)):
                gt_tokens = [
                    vocab[1][token_id]
                    for token_id in ground_truth_ids[i].tolist()
                    if token_id > 2
                ]
                gt_answer_string = " ".join(gt_tokens)
                pred_tokens = []
                for token_id in preds_ids[i]:
                    if token_id == 2:
                        break
                    if token_id > 2:
                        pred_tokens.append(vocab[1][token_id])
                pred_answer_string = " ".join(pred_tokens)

                gts["images"].append({"id": sample_id_counter})
                gts["annotations"].append(
                    {
                        "image_id": sample_id_counter,
                        "id": sample_id_counter,
                        "caption": gt_answer_string,
                    }
                )
                res.append(
                    {"image_id": sample_id_counter, "caption": pred_answer_string}
                )

                # --- NEPTUNE LOGGING (Samples) ---
                if run and sample_id_counter < 8:  # Log first 8 examples per eval
                    ref_img_np = batch["img_ref"][i].cpu().permute(1, 2, 0).numpy()
                    cur_img_np = batch["img_cur"][i].cpu().permute(1, 2, 0).numpy()

                    # Log images and text
                    run[f"val/samples/{sample_id_counter}/ref_image"].append(
                        File.as_image(ref_img_np)
                    )
                    run[f"val/samples/{sample_id_counter}/cur_image"].append(
                        File.as_image(cur_img_np)
                    )
                    run[f"val/samples/{sample_id_counter}/question"].append(qs[i])
                    run[f"val/samples/{sample_id_counter}/pred"].append(
                        pred_answer_string
                    )
                    run[f"val/samples/{sample_id_counter}/gt"].append(gt_answer_string)

                sample_id_counter += 1

    if res:
        print("\n--- Validation Metrics ---")
        val_scores = run_nlg_evaluation(gts, res)

        # --- NEPTUNE LOGGING (Metrics) ---
        if run and val_scores:
            for metric, score in val_scores.items():
                run[f"val/metrics/{metric}"].append(score)


def main(args):
    global logger
    if not logger.hasHandlers():
        logger = setup_logging(log_file="logs/train.log")

    # --- NEPTUNE INITIALIZATION ---
    global run
    run_name = f"topk-{args.topk}_lambda_cf-{args.lambda_cf}_dir-{args.ablation_no_direction}"
    try:
        run = neptune.init_run(
            project=NEPTUNE_PROJECT,
            name=run_name,
            api_token=NEPTUNE_API_TOKEN,
            source_files=[
                "src/train.py",
                "configs/clinicalbert_resnet.yaml",
                "lib/*.py",
                "lib/model/*.py",
            ],
        )
        run["parameters"] = vars(args)
        logger.info(f"Neptune run initialized: {run_name}")
    except Exception as e:
        logger.warning(f"Neptune init failed: {e}. Running without logging.")
        run = None
    # -----------------------------

    logger.info("Starting DRIFT-VQA Training.")
    logger.info(f"Full config: {vars(args)}")

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Data Loading ---
    logger.info(f"Loading training data from: {args.train_pairs_csv}")
    train_ds = DiffVQADataset(
        args.data_root,
        args.train_pairs_csv,
        args.train_meta_csv,
        split="train",
        max_ans_len=args.max_ans_len,
    )
    vocab = (train_ds.stoi, train_ds.itos)
    num_classes = len(train_ds.itos)

    vocab_save_path = Path("models/vocab.json")
    vocab_save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(vocab_save_path, "w") as f:
        json.dump({"stoi": train_ds.stoi, "itos": train_ds.itos}, f, indent=2)
    logger.info(f"Vocabulary saved to {vocab_save_path} (Size: {num_classes})")

    logger.info(f"Loading validation data from: {args.val_pairs_csv}")
    val_ds = DiffVQADataset(
        args.data_root,
        args.val_pairs_csv,
        args.val_meta_csv,
        split="val",
        vocab=vocab,
        max_ans_len=args.max_ans_len,
    )

    num_workers = args.num_workers
    logger.info(f"Using {num_workers} data loader workers.")
    train_loader = make_loader(train_ds, args.bs, shuffle=True, num_workers=num_workers)
    val_loader = make_loader(val_ds, args.bs, shuffle=False, num_workers=num_workers)

    # --- Model Initialization ---
    logger.info("Initializing model...")
    model = DiffVQAModel(
        backbone=args.backbone,
        text_encoder=args.text_encoder,
        text_model_name=args.text_model_name,
        text_dim=args.text_dim,
        text_proj_dim=args.text_proj_dim,
        text_finetune=args.text_finetune,
        topk=args.topk,
        num_classes=num_classes,
        max_ans_len=args.max_ans_len,
        mask_ratio=args.mask_ratio,
        ablation_no_direction=args.ablation_no_direction,  # <-- ABLATION FLAG
    ).to(device)

    if args.ckpt and Path(args.ckpt).exists():
        logger.info(f"Loading backbone weights from: {args.ckpt}")
        try:
            sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
            if "state_dict" in sd:
                sd = sd["state_dict"]
            elif "model" in sd:
                sd = sd["model"]

            missing, unexpected = model.drs.backbone.load_state_dict(sd, strict=False)
            logger.info(
                f"Loaded weights: missing={len(missing)} unexpected={len(unexpected)}"
            )
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")

    # --- Training Setup ---
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(args.lr),
        weight_decay=1e-4,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    print("\nStarting Model Training...")

    # --- Stage A (MRM) ---
    for ep in tqdm(
        range(int(args.epochs_mrm)), desc="Stage A (MRM)", leave=True, file=sys.stdout
    ):
        logger.info(f"\n=== Stage A Epoch {ep+1} ===")
        avg_loss = run_epoch(
            "mrm",
            model,
            train_loader,
            opt,
            scaler,
            device,
            main_loss_weight=args.main_loss_weight,
            lambda_mrm=1.0,
            lambda_align=0.0,
            lambda_cf=0.0,
            lambda_gate=0.0,
        )
        logger.info(f"Stage A Epoch {ep+1} Avg Loss: {avg_loss:.4f}")

    # --- Stage B (Warm-up) ---
    for ep in tqdm(
        range(int(args.epochs_warmup)),
        desc="Stage B (Warm-up)",
        leave=True,
        file=sys.stdout,
    ):
        logger.info(f"\n=== Stage B Epoch {ep+1} ===")
        avg_loss = run_epoch(
            "warmup",
            model,
            train_loader,
            opt,
            scaler,
            device,
            main_loss_weight=args.main_loss_weight,
            lambda_mrm=args.lambda_mrm,
            lambda_align=args.lambda_align,
            lambda_cf=0.0,
            lambda_gate=1e-3,
        )
        logger.info(f"Stage B Epoch {ep+1} Avg Loss: {avg_loss:.4f}")

    # --- RE-INITIALIZE OPTIMIZER FOR FINETUNING ---
    logger.info("Re-initializing optimizer for Stage C (VQA) with lower LR.")
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(args.lr),
        weight_decay=1e-4,
    )

    scheduler = CosineAnnealingLR(opt, T_max=int(args.epochs_vqa), eta_min=1e-8)

    # --- Stage C (VQA + Counterfactuals) ---
    for ep in tqdm(
        range(int(args.epochs_vqa)), desc="Stage C (VQA)", leave=True, file=sys.stdout
    ):
        logger.info(f"\n=== Stage C Epoch {ep+1} ===")
        avg_loss = run_epoch(
            "vqa",
            model,
            train_loader,
            opt,
            scaler,
            device,
            main_loss_weight=args.main_loss_weight,
            lambda_mrm=0.0,
            lambda_align=0.0,
            lambda_cf=args.lambda_cf,  # Pass the config value
            lambda_gate=1e-3,
        )
        logger.info(f"Stage C Epoch {ep+1} Avg Loss: {avg_loss:.4f}")

        scheduler.step()

        logger.info(f"current LR: {opt.param_groups[0]['lr']:.8f}")

        logger.info("Running validation...")
        evaluate(model, val_loader, device, vocab)

    # --- Save Model ---
    save_path = Path(
        f"models/drift_k-{args.topk}_lambda_cf-{args.lambda_cf}_nodir-{args.ablation_no_direction}.pth"
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    logger.info(f"Training complete. Model saved to {save_path}")

    if run:
        run.stop()


if __name__ == "__main__":
    Path("logs").mkdir(exist_ok=True)
    logger = setup_logging(log_file="logs/train.log", console_level=logging.INFO)

    try:
        args = parse_args()
        main(args)
    except Exception as e:
        logger.exception("Training failed due to an uncaught exception:")
        raise e


# models/drift_k-{args.topk}_lambda_cf-{args.lambda_cf}_dir-{args.no_direction}.pth
