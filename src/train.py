# train.py
import argparse, random, json, yaml
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler
from torch.optim import AdamW

from lib.dataset import DiffVQADataset
from lib.utils import make_loader, setup_logging, run_nlg_evaluation
from lib.model import DiffVQAModel
from lib.losses import heatmap_kl, info_nce_token_sets
from lib.phrases import load_keyinfo
from lib.negate import negate_question

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log_file = "logs/train.log"
logger = setup_logging(log_file=log_file)


# --------------------------
# Args + YAML config
# --------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="", help="Path to YAML config")

    # All CLI args are OPTIONAL; YAML can override them
    # Data
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--pairs_csv", type=str, default="")
    parser.add_argument("--meta_csv", type=str, default="")
    parser.add_argument("--keyinfo_json", type=str, default="")
    parser.add_argument("--ckpt", type=str, default="")

    # Model
    parser.add_argument("--backbone", type=str, default="")
    # parser.add_argument(
    #     "--head", type=str, default="classifier", choices=["classifier", "decoder"]
    # )

    # Text encoder
    parser.add_argument(
        "--text_encoder", type=str, default="tiny", choices=["tiny", "clinicalbert"]
    )
    parser.add_argument(
        "--text_model_name", type=str, default="emilyalsentzer/Bio_ClinicalBERT"
    )
    parser.add_argument("--text_finetune", action="store_true")
    parser.add_argument("--text_dim", type=int, default=768)  # ClinicalBERT hidden size
    parser.add_argument(
        "--text_proj_dim", type=int, default=256
    )  # projected dim into QDT

    # Decoder vocab (when head=decoder)
    parser.add_argument("--dec_vocab", type=int, default=6000)

    # Train hyperparams
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs_mrm", type=int, default=1)
    parser.add_argument("--epochs_warmup", type=int, default=1)
    parser.add_argument("--epochs_vqa", type=int, default=3)
    parser.add_argument("--topk", type=int, default=64)

    # Loss weights
    parser.add_argument("--lambda_mrm", type=float, default=0.1)
    parser.add_argument("--lambda_align", type=float, default=0.05)
    parser.add_argument("--lambda_cf", type=float, default=0.05)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # YAML overrides CLI defaults
    if args.config and Path(args.config).exists():
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
        for k, v in cfg.items():
            setattr(args, k, v)

    # Basic sanity: these should be set by YAML or CLI
    needed = ["data_root", "pairs_csv", "meta_csv"]
    missing = [k for k in needed if not getattr(args, k, None)]
    if missing:
        raise SystemExit(
            f"Missing required settings ({', '.join(missing)}). "
            f"Provide them in the YAML passed by --config or as CLI flags."
        )

    return args


# --------------------------
# Utils
# --------------------------
def seed_all(s=42):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


def text_to_ids(texts, vocab_size=6000, max_len=16):
    """Hash-based toy tokenizer for decoder targets; 0=PAD."""
    ids = []
    for t in texts:
        words = t.strip().lower().split()[:max_len] or ["<blank>"]
        row = [(hash(w) % (vocab_size - 1)) + 1 for w in words]
        row += [0] * (max_len - len(row))
        ids.append(row)
    return torch.tensor(ids, dtype=torch.long)


def tokenize_questions(text_model, batch_questions, use_hf=False, device=None):
    """Return token ids (TinyText) or dict of tensors (HF ClinicalBERT)."""
    if use_hf:
        enc = text_model.tokenize(batch_questions)
        if device is not None:
            enc = {k: v.to(device) for k, v in enc.items()}
        return enc
    else:
        # create one tensor and immediately move it to the correct device
        token_ids = text_model.tokenize(batch_questions)
        if device is not None:
            token_ids = token_ids.to(device)
        return token_ids

        # return text_model.tokenize(batch_questions)


# --------------------------
# One epoch
# --------------------------
def run_epoch(
    stage,
    model: DiffVQAModel,
    loader,
    optimizer: AdamW,
    scaler: GradScaler,
    device,
    lambda_mrm=0.1,
    lambda_align=0.05,
    lambda_cf=0.05,
    lambda_gate=1e-3,
):
    model.train()
    total_steps = len(loader)
    running = {"loss": 0.0, "acc": 0, "n": 0}

    for i, batch in enumerate(loader):
        img_cur = batch["img_cur"].to(device)
        img_ref = batch["img_ref"].to(device)
        qs = batch["question"]
        qs_cf = [negate_question(q) for q in qs]

        y_seq = batch["answer_ids"].to(device)

        tokens = tokenize_questions(
            model.text, qs, use_hf=getattr(model, "uses_hf", False), device=device
        )
        tokens_cf = tokenize_questions(
            model.text, qs_cf, use_hf=getattr(model, "uses_hf", False), device=device
        )
        img_cur_cf, img_ref_cf = img_ref, img_cur

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            out = model(img_ref, img_cur, tokens)
            out_cf = model(img_ref_cf, img_cur_cf, tokens_cf)

            loss_mrm = out["loss_mrm"]
            loss_align = model.drs.alignment_loss(
                out["r_pos"], out["r_neg"], out["r_abs"], out["signed"]
            )
            loss_hkl = heatmap_kl(out["heatmap"], out_cf["heatmap"])
            loss_nce = info_nce_token_sets(out["patches"], out_cf["patches"])
            loss_gate = out["gate_l1"]

            main_loss = 0

            _, main_loss = model.head(out["sel_tokens"], targets=y_seq)

            if stage == "mrm":
                loss = loss_mrm
                logger.info(
                    f"  [Step {i+1}/{total_steps}] MRM Loss: {loss_mrm.item():.4f}"
                )

            elif stage == "warmup":
                loss = (
                    main_loss
                    + lambda_mrm * loss_mrm
                    + lambda_align * loss_align
                    + lambda_gate * loss_gate
                )
                logger.debug(
                    f"  [Step {i+1}/{total_steps}] Warmup Losses -> "
                    f"Main: {main_loss.item():.4f}, MRM: {loss_mrm.item():.4f}, Align: {loss_align.item():.4f}"
                )

            else:  # stage == "vqa"
                loss_cf_combined = loss_hkl + loss_nce
                loss = (
                    main_loss
                    + lambda_mrm * loss_mrm
                    + lambda_align * loss_align
                    + lambda_cf * loss_cf_combined
                    + lambda_gate * loss_gate
                )
                logger.debug(
                    f"  [Step {i+1}/{total_steps}] VQA Losses -> "
                    f"Main: {main_loss.item():.4f}, MRM: {loss_mrm.item():.4f}, Align: {loss_align.item():.4f}, CF: {loss_cf_combined.item():.4f}"
                )

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        running["loss"] += loss.item()

        if (i + 1) % 50 == 0:
            logger.info(
                f"[{stage}] {i+1}/{total_steps} loss={running['loss']/(i+1):.4f}"
            )


def evaluate(model, loader, device, vocab):
    model.eval()  # Set the model to evaluation mode

    # --- Decoder Evaluation Logic ---
    gts = {"info": {}, "images": [], "annotations": []}
    res = []
    sample_id_counter = 0

    with torch.no_grad():  # Disable gradient calculation
        for batch in loader:
            img_cur = batch["img_cur"].to(device)
            img_ref = batch["img_ref"].to(device)
            qs = batch["question"]
            ground_truth_ids = batch["answer_ids"].cpu()  # Get ground truth sequences

            tokens = tokenize_questions(
                model.text, qs, use_hf=getattr(model, "uses_hf", False), device=device
            )
            out = model(img_ref, img_cur, tokens)

            # Generate predictions
            _, preds_ids = model.head(out["sel_tokens"])
            preds_ids = preds_ids.cpu().tolist()

            # Collect ground truths and predictions
            for i in range(len(qs)):
                # Convert ground truth IDs to string
                gt_tokens = [
                    vocab[1][token_id]
                    for token_id in ground_truth_ids[i].tolist()
                    if token_id > 2
                ]
                gt_answer_string = " ".join(gt_tokens)

                # Convert predicted IDs to string
                pred_tokens = []
                for token_id in preds_ids[i]:
                    if token_id == 2:
                        break  # Stop at <end> token
                    if token_id > 2:
                        pred_tokens.append(vocab[1][token_id])
                pred_answer_string = " ".join(pred_tokens)

                # Populate COCO-style dictionaries
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
                sample_id_counter += 1

    # Run the NLG evaluation
    if res:
        print("\n--- Validation Metrics ---")
        run_nlg_evaluation(gts, res)


# --------------------------
# Main
# --------------------------
def main(args):

    logger.info("Starting DRIFT-VQA Training")
    logger.info(f"Configuration: {vars(args)}")

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {torch.cuda.get_device_name()}")

    # datasets (ensure your DiffVQADataset filters by split internally)
    train_ds = DiffVQADataset(
        args.data_root,
        args.train_pairs_csv,
        args.train_meta_csv,
        split="train",
        max_ans_len=args.max_ans_len,
    )
    vocab = (train_ds.stoi, train_ds.itos)
    # num_classes = len(train_ds.itos) if args.head == "classifier" else args.dec_vocab
    num_classes = len(train_ds.itos)

    vocab_save_path = Path("models/vocab.json")
    vocab_save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(vocab_save_path, "w") as f:
        json.dump({"stoi": train_ds.stoi, "itos": train_ds.itos}, f, indent=4)
    logger.info(f"vocabulary saved to {vocab_save_path}")

    val_ds = DiffVQADataset(
        args.data_root,
        args.val_pairs_csv,
        args.val_meta_csv,
        split="val",
        vocab=vocab,
        max_ans_len=args.max_ans_len,
    )

    print(f"Answer classes / Decoder vocab: {num_classes}")

    train_loader = make_loader(train_ds, args.bs, shuffle=True)
    val_loader = make_loader(
        val_ds, args.bs, shuffle=False
    )  # TODO: wire evaluation if needed

    # Model
    model = DiffVQAModel(
        backbone=args.backbone,
        text_encoder=args.text_encoder,
        text_model_name=args.text_model_name,
        text_dim=args.text_dim,
        text_proj_dim=args.text_proj_dim,
        text_finetune=args.text_finetune,
        topk=args.topk,
        num_rows=3,
        num_cols=2,
        num_classes=num_classes,
        max_ans_len=args.max_ans_len,
    ).to(device)

    opt = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(args.lr),
        weight_decay=1e-4,
    )

    scaler = GradScaler(enabled=(device.type == "cuda"))

    # stage A — MaskedResidualModel warm-up
    for ep in range(int(args.epochs_mrm)):
        print(f"\n=== Stage A: MaskedResidualModel epoch {ep+1}/{args.epochs_mrm} ===")
        run_epoch(
            "mrm",
            model,
            train_loader,
            opt,
            scaler,
            device,
            lambda_mrm=1.0,
            lambda_align=0.0,
            lambda_cf=0.0,
            lambda_gate=0.0,
        )

    # stage B — warm-up with KeyInfo phrases
    for ep in range(int(args.epochs_warmup)):
        print(f"\n=== Stage B: Warm-up epoch {ep+1}/{args.epochs_warmup} ===")
        run_epoch(
            "warmup",
            model,
            train_loader,
            opt,
            scaler,
            device,
            lambda_mrm=0.1,
            lambda_align=0.05,
            lambda_cf=0.0,
            lambda_gate=1e-3,
        )

    # stage C — Diff-VQA finetune with counterfactual evidence losses
    for ep in range(int(args.epochs_vqa)):
        print(f"\n=== Stage C: VQA epoch {ep+1}/{args.epochs_vqa} ===")
        run_epoch(
            "vqa",
            model,
            train_loader,
            opt,
            scaler,
            device,
            lambda_mrm=args.lambda_mrm,
            lambda_align=args.lambda_align,
            lambda_cf=args.lambda_cf,
            lambda_gate=1e-3,
        )

        print("Running validation...")
        evaluate(model, val_loader, device, vocab)

    # --- SAVE THE FINAL MODEL ---
    print("\nTraining complete. Saving final model...")
    save_path = Path("models/drift_vqa_final.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
