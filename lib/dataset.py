# dataset.py
import csv, json
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def gray_to_rgb(img):
    if img.mode != "L":
        img = img.convert("L")
    return img.convert("RGB")


img_tf = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Lambda(gray_to_rgb),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


class DiffVQADataset(Dataset):
    """
    Uses mimic_all.csv (to resolve study_id->jpg) and mimic_pair_questions.csv (pairs + QA).
    Builds closed answer vocab from the train split if not provided.
    """

    def __init__(self, data_root, pairs_csv, meta_csv, split="train", vocab=None):
        self.data_root = Path(data_root)
        self.rows = []
        self.study_to_path = self._build_map(meta_csv)

        with open(pairs_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r["split"] != split:
                    continue
                if (
                    r["study_id"] in self.study_to_path
                    and r["ref_id"] in self.study_to_path
                ):
                    self.rows.append(r)

        if vocab is None:
            answers = [self._norm(r["answer"]) for r in self.rows]
            itos = sorted(Counter(answers).keys())
            self.stoi = {
                t: i + 1 for i, t in enumerate(itos)
            }  # 0 reserved for <pad>/<unk>
            self.itos = ["<pad>"] + itos
        else:
            self.stoi, self.itos = vocab

    def _build_map(self, meta_csv):
        m = {}
        with open(meta_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row["study_id"]
                subj = row["subject_id"]
                pfx = f"p{str(subj)[:2]}"
                pdir = self.data_root / pfx / f"p{subj}" / f"s{sid}"
                jpgs = sorted(pdir.glob("*.jpg"))
                if jpgs:
                    m[sid] = str(jpgs[0])
        return m

    def _norm(self, s):
        return s.strip().lower().replace(".", "").replace(",", "")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        cur_path = self.study_to_path[r["study_id"]]
        ref_path = self.study_to_path[r["ref_id"]]
        q = r["question"].strip().lower()
        a = self._norm(r["answer"])
        y = self.stoi.get(a, 0)

        img_cur = img_tf(Image.open(cur_path))
        img_ref = img_tf(Image.open(ref_path))

        return {
            "img_cur": img_cur,
            "img_ref": img_ref,
            "question": q,
            "answer_id": torch.tensor(y, dtype=torch.long),  # 0 can be <unk>
            "meta": (r["subject_id"], r["study_id"], r["ref_id"]),
        }
