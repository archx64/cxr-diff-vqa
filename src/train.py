# train.py
import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from lib.dataset import DiffVQADataset
from lib.utils import make_loader
from lib.models import (
    DirectionalResidualStack,
    QuestionGuidedDifferenceTokenizer,
    MaskedResidualModel,
    IDEClassifier,
    TinyTransformerDecoder,
    TinyText,
    ClinicalBERTText,
)
from lib.losses import heatmap_kl, info_nce_token_sets
from lib.phrases import load_keyinfo, build_diff_phrase
from lib.negate import negate_question
from lib.utils import setup_logging

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log_file = 'logs/train.log'
logger = setup_logging(log_file=log_file, logger_name=__name__)

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
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument(
        "--head", type=str, default="classifier", choices=["classifier", "decoder"]
    )

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
# Model wrapper
# --------------------------
class DiffVQAModel(nn.Module):
    def __init__(
        self,
        backbone="resnet50",
        text_encoder="tiny",
        text_model_name="emilyalsentzer/Bio_ClinicalBERT",
        text_dim=768,
        text_proj_dim=256,
        text_finetune=False,
        topk=64,
        num_rows=3,
        num_cols=2,
        num_classes=1000,
        head="classifier",
    ):
        super().__init__()

        # Vision encoder (DRS+)
        self.drs = DirectionalResidualStack(backbone_name=backbone)
        C = self.drs.out_channels
        c_all = C * 4  # [R+, R-, Rabs, signed]

        # Text encoder
        if text_encoder == "clinicalbert":
            self.text = ClinicalBERTText(
                model_name=text_model_name,
                d_txt=text_dim,
                proj_dim=text_proj_dim,
                fine_tune=text_finetune,
            )
            self.uses_hf = True
            q_dim = text_proj_dim
        else:
            self.text = TinyText(d_txt=text_proj_dim)
            self.uses_hf = False
            q_dim = text_proj_dim

        # QDT+ and MaskedResidualModel
        self.qdt = QuestionGuidedDifferenceTokenizer(
            c_img=c_all, d_txt=q_dim, k=topk, num_rows=num_rows, num_cols=num_cols
        )
        self.mrm = MaskedResidualModel(c_all=c_all, mask_ratio=0.6)

        # Head
        if head == "classifier":
            self.head = IDEClassifier(dim=c_all, num_classes=num_classes)
            self.is_classifier = True
        else:
            self.head = TinyTransformerDecoder(
                dim=c_all, vocab_size=num_classes, nlayer=3, nhead=8, max_len=16
            )
            self.is_classifier = False

    def forward(self, img_ref, img_cur, token_batch):
        # Visual residuals
        r = self.drs(img_ref, img_cur)  # dict: r_pos, r_neg, r_abs, signed

        # Question vector
        q_vec = self.text(token_batch)  # (B, q_dim)

        # Token selection + heatmap + gate sparsity
        sel_tokens, heatmap, gate_l1 = self.qdt(
            q_vec, r
        )  # (B,k,c_all), (B,H,W), scalar

        # MaskedResidualModel on residual token maps
        feats_for_mrm = torch.cat(
            [r["r_pos"], r["r_neg"], r["r_abs"], r["signed"]], dim=1
        )
        mrm_out = self.mrm(feats_for_mrm)

        if self.is_classifier:
            logits = self.head(sel_tokens, token_kinds=None)
            return {
                "logits": logits,
                "heatmap": heatmap,
                "gate_l1": gate_l1,
                **mrm_out,
                **r,
            }
        else:
            return {
                "sel_tokens": sel_tokens,
                "heatmap": heatmap,
                "gate_l1": gate_l1,
                **mrm_out,
                **r,
            }


# --------------------------
# One epoch
# --------------------------
def run_epoch(
    stage,
    model,
    loader,
    optimizer,
    scaler,
    device,
    lambda_mrm=0.1,
    lambda_align=0.05,
    lambda_cf=0.05,
    lambda_gate=1e-3,
    classifier=True,
    vocab_size=None,
    keyinfo_idx=None,
):
    model.train()
    total_steps = len(loader)
    running = {"loss": 0.0, "acc": 0, "n": 0}

    for i, batch in enumerate(loader):
        img_cur = batch["img_cur"].to(device)
        img_ref = batch["img_ref"].to(device)
        y = batch["answer_id"].to(device)  # classifier path
        qs = [q for q in batch["question"]]
        qs_cf = [negate_question(q) for q in qs]

        tokens = tokenize_questions(
            model.text, qs, use_hf=getattr(model, "uses_hf", False), device=device
        )

        if isinstance(tokens, torch.Tensor):
            tokens = tokens.to(device=device, dtype=torch.long)

        tokens_cf = tokenize_questions(
            model.text, qs_cf, use_hf=getattr(model, "uses_hf", False), device=device
        )

        # Counterfactual image swap
        img_cur_cf, img_ref_cf = img_ref, img_cur

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            out = model(img_ref, img_cur, tokens)
            out_cf = model(img_ref_cf, img_cur_cf, tokens_cf)

            # Base components
            loss_mrm = out["loss_mrm"]
            loss_align = model.drs.alignment_loss(
                out["r_pos"], out["r_neg"], out["r_abs"], out["signed"]
            )
            loss_hkl = heatmap_kl(out["heatmap"], out_cf["heatmap"])
            loss_nce = info_nce_token_sets(out["patches"], out_cf["patches"])
            loss_gate = out["gate_l1"]

            if stage == "mrm":
                loss = loss_mrm
                logger.info(f"  [Step {i+1}/{total_steps}] MRM Loss: {loss_mrm.item():.4f}")

            elif stage == "warmup":
                if classifier:
                    logits = out["logits"]
                    ce = F.cross_entropy(logits, y, ignore_index=0)
                    loss = (
                        ce
                        + lambda_mrm * loss_mrm
                        + lambda_align * loss_align
                        + lambda_gate * loss_gate
                    )
                else:
                    # Generative warm-up via KeyInfo deltas (if available)
                    phrases = []
                    for m in batch["meta"]:
                        _, sid_cur, sid_ref = m
                        phrases.append(
                            build_diff_phrase(sid_cur, sid_ref, keyinfo_idx)
                            if keyinfo_idx
                            else "no significant change"
                        )
                    targets = text_to_ids(
                        phrases, vocab_size=vocab_size, max_len=16
                    ).to(device)
                    _, loss_dec = model.head(
                        out["sel_tokens"], targets=targets, token_kinds=None
                    )
                    loss = (
                        loss_dec
                        + lambda_mrm * loss_mrm
                        + lambda_align * loss_align
                        + lambda_gate * loss_gate
                    )
                    logger.debug(
                        f"  [Step {i+1}/{total_steps}] Warmup Losses -> "
                        f"CE: {ce.item():.4f}, MRM: {loss_mrm.item():.4f}, Align: {loss_align.item():.4f}"
                    )

            else:  # stage == "vqa"
                if classifier:
                    logits = out["logits"]
                    ce = F.cross_entropy(logits, y, ignore_index=0)
                    loss_cf_combined = loss_hkl + loss_nce
                    loss = (
                        ce
                        + lambda_mrm * loss_mrm
                        + lambda_align * loss_align
                        + lambda_cf * (loss_hkl + loss_nce)
                        + lambda_gate * loss_gate
                    )
                    logger.debug(
                        f"  [Step {i+1}/{total_steps}] VQA Losses -> "
                        f"CE: {ce.item():.4f}, MRM: {loss_mrm.item():.4f}, Align: {loss_align.item():.4f}, CF: {loss_cf_combined.item():.4f}"
                    )
                else:
                    # Optionally continue supervising with KeyInfo phrases
                    phrases = []
                    for m in batch["meta"]:
                        _, sid_cur, sid_ref = m
                        phrases.append(
                            build_diff_phrase(sid_cur, sid_ref, keyinfo_idx)
                            if keyinfo_idx
                            else "no significant change"
                        )
                    targets = text_to_ids(
                        phrases, vocab_size=vocab_size, max_len=16
                    ).to(device)
                    _, loss_dec = model.head(
                        out["sel_tokens"], targets=targets, token_kinds=None
                    )
                    loss = (
                        loss_dec
                        + lambda_mrm * loss_mrm
                        + lambda_align * loss_align
                        + lambda_cf * (loss_hkl + loss_nce)
                        + lambda_gate * loss_gate
                    )

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        running["loss"] += loss.item()
        if classifier:
            with torch.no_grad():
                pred = out["logits"].argmax(dim=-1)
                mask = y != 0
                running["acc"] += (pred[mask] == y[mask]).sum().item()
                running["n"] += mask.sum().item()

        if (i + 1) % 50 == 0:
            if classifier and running["n"] > 0:
                logger.info(
                    f"[{stage}] {i+1}/{total_steps} "
                    f"loss={running['loss']/(i+1):.4f} acc={running['acc']/max(1,running['n']):.3f}"
                )
            else:
                logger.info(f"[{stage}] {i+1}/{total_steps} loss={running['loss']/(i+1):.4f}")


def evaluate(model, loader, device, classifier=True):
    model.eval()  # Set the model to evaluation mode
    running = {"acc": 0, "n": 0}

    with torch.no_grad():  # Disable gradient calculation
        for i, batch in enumerate(loader):
            img_cur = batch["img_cur"].to(device)
            img_ref = batch["img_ref"].to(device)
            y = batch["answer_id"].to(device)
            qs = [q for q in batch["question"]]

            tokens = tokenize_questions(
                model.text, qs, use_hf=getattr(model, "uses_hf", False), device=device
            )

            # Only need a forward pass, no counterfactuals needed for evaluation
            out = model(img_ref, img_cur, tokens)

            if classifier:
                pred = out["logits"].argmax(dim=-1)
                mask = y != 0  # Ignore padding/unknown answers
                running["acc"] += (pred[mask] == y[mask]).sum().item()
                running["n"] += mask.sum().item()
            else:
                # Evaluation for a decoder is more complex (BLEU, CIDEr etc.)
                # For now, we'll just report accuracy on the classifier head
                pass

    if classifier and running["n"] > 0:
        accuracy = running["acc"] / running["n"]
        print(f"\nValidation Accuracy: {accuracy:.4f}\n")
        return accuracy
    return 0.0


# --------------------------
# Main
# --------------------------
def main(args):

    logger.info("Starting DRIFT-VQA Training")
    logger.info(f"Configuration: {vars(args)}")

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {torch.get_di}")

    # Datasets (ensure your DiffVQADataset filters by split internally)
    train_ds = DiffVQADataset(
        args.data_root, args.pairs_csv, args.meta_csv, split="train"
    )
    vocab = (train_ds.stoi, train_ds.itos)
    val_ds = DiffVQADataset(
        args.data_root, args.pairs_csv, args.meta_csv, split="val", vocab=vocab
    )

    num_classes = len(train_ds.itos) if args.head == "classifier" else args.dec_vocab
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
        head=args.head,
    ).to(device)

    # Optional: load CXR-CLIP / SwinTiny weights
    if args.ckpt and Path(args.ckpt).exists():
        sd = torch.load(args.ckpt, map_location="cpu")
        missing, unexpected = model.drs.backbone.load_state_dict(sd, strict=False)
        print(
            f"Loaded backbone weights: missing={len(missing)} unexpected={len(unexpected)}"
        )

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(args.lr),
        weight_decay=1e-4,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # keyInfo index for phrase supervision
    keyinfo_idx = (
        load_keyinfo(args.keyinfo_json)
        if args.keyinfo_json and Path(args.keyinfo_json).exists()
        else None
    )

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
            classifier=(args.head == "classifier"),
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
            classifier=(args.head == "classifier"),
            vocab_size=(args.dec_vocab if args.head == "decoder" else None),
            keyinfo_idx=keyinfo_idx,
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
            classifier=(args.head == "classifier"),
            vocab_size=(args.dec_vocab if args.head == "decoder" else None),
            keyinfo_idx=keyinfo_idx,
        )
        
        print("Running validation...")
        evaluate(model, val_loader, device, classifier=(args.head == "classifier"))


    # --- SAVE THE FINAL MODEL ---
    print("\nTraining complete. Saving final model...")
    save_path = Path("models/drift_vqa_final.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
