# train.py
import argparse, random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from lib.dataset import DiffVQADataset
from lib.models import (
    DirectionalResidualStack,
    QuestionGuidedDifferenceTokenizer,
    TinyText,
    MaskedResidualModel,
    IDEClassifier,
    TinyTransformerDecoder,
)


from lib.phrases import load_keyinfo, build_diff_phrase
from lib.negate import negate_question
from lib.losses import heatmap_kl, info_nce_token_sets


def heatmap_kl(h1, h2, eps=1e-6):
    # KL(P||Q) + KL(Q||P) symmetric
    p = h1.float() + eps
    q = h2.float() + eps
    p = p / p.sum(dim=(1, 2), keepdim=True)
    q = q / q.sum(dim=(1, 2), keepdim=True)
    kl1 = (p * (p.log() - q.log())).sum(dim=(1, 2))
    kl2 = (q * (q.log() - p.log())).sum(dim=(1, 2))
    return (kl1 + kl2).mean()


def info_nce_token_sets(toks_a, toks_b, temperature=0.07):
    """
    toks_*: (B,k,D) — mean-pool and contrast batch-wise.
    """
    a = toks_a.mean(dim=1)  # (B,D)
    b = toks_b.mean(dim=1)
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    logits = a @ b.t() / temperature
    labels = torch.arange(a.size(0), device=a.device)
    return F.cross_entropy(logits, labels)


def collate(batch):
    # simple default collator (images are same size from transforms)
    out = {k: [] for k in batch[0].keys()}
    for b in batch:
        for k, v in b.items():
            out[k].append(v)
    out["img_cur"] = torch.stack(out["img_cur"])
    out["img_ref"] = torch.stack(out["img_ref"])
    out["answer_id"] = torch.stack(out["answer_id"])
    return out


def make_loader(ds, bs, shuffle):
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate,
    )


def seed_all(s=69):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


class DiffVQAModel(nn.Module):
    def __init__(
        self,
        backbone="resnet50",
        txt_dim=256,
        topk=64,
        num_rows=3,
        num_cols=2,
        num_classes=1000,
        head="classifier",
    ):
        super().__init__()
        self.drs = DirectionalResidualStack(backbone_name=backbone)
        C = self.drs.out_channels
        self.text = TinyText(d_txt=txt_dim)
        self.qdt = QuestionGuidedDifferenceTokenizer(
            c_img=(C * 4), d_txt=txt_dim, k=topk, num_rows=num_rows, num_cols=num_cols
        )
        self.mrm = MaskedResidualModel(c_all=(C * 4), mask_ratio=0.6)
        if head == "classifier":
            self.head = IDEClassifier(dim=(C * 4), num_classes=num_classes)
            self.is_classifier = True
        else:
            self.head = TinyTransformerDecoder(
                dim=(C * 4), vocab_size=num_classes, nlayer=3, nhead=8, max_len=16
            )
            self.is_classifier = False

    def forward(self, img_ref, img_cur, token_ids):
        r = self.drs(img_ref, img_cur)
        q_vec = self.text(token_ids)
        sel_tokens, heatmap, gate_l1 = self.qdt(q_vec, r)
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


def tokenize_questions(text_model, batch_questions):
    with torch.no_grad():
        return text_model.tokenize(batch_questions)


def text_to_ids(texts, vocab_size=6000, max_len=16):
    ids = []
    for t in texts:
        words = t.strip().lower().split()[:max_len]
        if not words:
            words = ["<blank>"]
        row = [(hash(w) % (vocab_size - 1)) + 1 for w in words]  # 0=pad
        row += [0] * (max_len - len(row))
        ids.append(row)
    return torch.tensor(ids, dtype=torch.long)


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
        y = batch["answer_id"].to(device)
        qs = [q for q in batch["question"]]
        qs_cf = [negate_question(q) for q in qs]

        token_ids = tokenize_questions(model.text, qs).to(device)
        token_ids_cf = tokenize_questions(model.text, qs_cf).to(device)

        img_cur_cf, img_ref_cf = img_ref, img_cur

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            out = model(img_ref, img_cur, token_ids)
            out_cf = model(img_ref_cf, img_cur_cf, token_ids_cf)

            loss_mrm = out["loss_mrm"]
            loss_align = model.drs.alignment_loss(
                out["r_pos"], out["r_neg"], out["r_abs"], out["signed"]
            )
            loss_hkl = heatmap_kl(out["heatmap"], out_cf["heatmap"])
            loss_nce = info_nce_token_sets(out["patches"], out_cf["patches"])
            loss_gate = out["gate_l1"]

            if stage == "mrm":
                loss = loss_mrm
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
                    phrases = []
                    for m in batch["meta"]:
                        _, sid_cur, sid_ref = m
                        phrases.append(
                            build_diff_phrase(sid_cur, sid_ref, keyinfo_idx)
                            if keyinfo_idx
                            else "no change"
                        )
                    targets = text_to_ids(
                        phrases, vocab_size=vocab_size, max_len=16
                    ).to(device)
                    logits, loss_dec = model.head(
                        out["sel_tokens"], targets=targets, token_kinds=None
                    )
                    loss = (
                        loss_dec
                        + lambda_mrm * loss_mrm
                        + lambda_align * loss_align
                        + lambda_gate * loss_gate
                    )
            else:  # vqa
                if classifier:
                    logits = out["logits"]
                    ce = F.cross_entropy(logits, y, ignore_index=0)
                    loss = (
                        ce
                        + lambda_mrm * loss_mrm
                        + lambda_align * loss_align
                        + lambda_cf * (loss_hkl + loss_nce)
                        + lambda_gate * loss_gate
                    )
                else:
                    phrases = []
                    for m in batch["meta"]:
                        _, sid_cur, sid_ref = m
                        phrases.append(
                            build_diff_phrase(sid_cur, sid_ref, keyinfo_idx)
                            if keyinfo_idx
                            else "no change"
                        )
                    targets = text_to_ids(
                        phrases, vocab_size=vocab_size, max_len=16
                    ).to(device)
                    logits, loss_dec = model.head(
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
                print(
                    f"[{stage}] {i+1}/{total_steps} loss={running['loss']/(i+1):.4f} acc={running['acc']/max(1,running['n']):.3f}"
                )
            else:
                print(f"[{stage}] {i+1}/{total_steps} loss={running['loss']/(i+1):.4f}")


def main(args):
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = DiffVQADataset(
        args.data_root, args.pairs_csv, args.meta_csv, split="train"
    )
    vocab = (train_ds.stoi, train_ds.itos)
    val_ds = DiffVQADataset(
        args.data_root, args.pairs_csv, args.meta_csv, split="val", vocab=vocab
    )

    num_classes = len(train_ds.itos) if args.head == "classifier" else args.dec_vocab
    print(f"Answer classes: {num_classes}")

    train_loader = make_loader(train_ds, args.bs, shuffle=True)
    val_loader = make_loader(val_ds, args.bs, shuffle=False)

    model = DiffVQAModel(
        backbone=args.backbone,
        txt_dim=256,
        topk=args.topk,
        num_rows=3,
        num_cols=2,
        num_classes=num_classes,
        head=args.head,
    ).to(device)

    if args.ckpt and Path(args.ckpt).exists():
        sd = torch.load(args.ckpt, map_location="cpu")
        missing, unexpected = model.drs.backbone.load_state_dict(sd, strict=False)
        print(
            f"Loaded backbone weights: missing={len(missing)} unexpected={len(unexpected)}"
        )

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    keyinfo_idx = (
        load_keyinfo(args.keyinfo_json)
        if args.keyinfo_json and Path(args.keyinfo_json).exists()
        else None
    )

    for ep in range(args.epochs_mrm):
        print(f"\n=== Stage A: MRM epoch {ep+1}/{args.epochs_mrm} ===")
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

    for ep in range(args.epochs_warmup):
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

    for ep in range(args.epochs_vqa):
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


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--pairs_csv", type=str, required=True)
    p.add_argument("--meta_csv", type=str, required=True)
    p.add_argument("--keyinfo_json", type=str, default="")
    p.add_argument("--ckpt", type=str, default="")
    p.add_argument("--backbone", type=str, default="resnet50")
    p.add_argument(
        "--head", type=str, default="classifier", choices=["classifier", "decoder"]
    )
    p.add_argument("--dec_vocab", type=int, default=6000)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs_mrm", type=int, default=1)
    p.add_argument("--epochs_warmup", type=int, default=1)
    p.add_argument("--epochs_vqa", type=int, default=3)
    p.add_argument("--topk", type=int, default=64)
    p.add_argument("--lambda_mrm", type=float, default=0.1)
    p.add_argument("--lambda_align", type=float, default=0.05)
    p.add_argument("--lambda_cf", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)
