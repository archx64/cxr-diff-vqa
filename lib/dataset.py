# dataset.py
import csv, logging
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
        transforms.Resize((384, 384)),
        transforms.Lambda(gray_to_rgb),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

logger = logging.getLogger(__name__)



class DiffVQADataset(Dataset):
    def __init__(
        self, data_root, pairs_csv, meta_csv, split="train", vocab=None, max_ans_len=32
    ):
        self.data_root = Path(data_root)
        self.max_ans_len = max_ans_len
        cache_file = Path(data_root) / f"mimic_diff_vqa_{split}_word_cache.pt"

        if cache_file.exists():
            logger.info(f"Loading cached dataset manifest from {cache_file}...")
            cached_data = torch.load(cache_file)
            self.rows = cached_data["rows"]
            self.study_to_path = cached_data["study_to_path"]
        else:
            logger.info(
                f"No word-level cache found for '{split}'. Building manifest from CSVs..."
            )
            self.rows = []
            logger.info(
                f"[{split} split] Building study_to_path map from {meta_csv}..."
            )
            self.study_to_path = self._build_map(meta_csv)
            logger.info(
                f"Map built. Found paths for {len(self.study_to_path)} studies."
            )

            logger.info(
                f"[{split} split] Loading all question pairs from pre-split file: {pairs_csv}..."
            )
            with open(pairs_csv, "r", newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if (
                        r["study_id"] in self.study_to_path
                        and r["ref_id"] in self.study_to_path
                    ):
                        self.rows.append(r)

            logger.info(f"[{split} split] Loaded {len(self.rows)} question pairs.")
            logger.info(f"Saving dataset manifest to {cache_file} for future runs...")
            torch.save(
                {"rows": self.rows, "study_to_path": self.study_to_path}, cache_file
            )

        if vocab is None:
            logger.info(f"[{split} split] Building new WORD-LEVEL vocabulary...")
            all_answers_text = [self._norm(r["answer"]) for r in self.rows]
            word_counts = Counter(
                word for text in all_answers_text for word in text.split()
            )

            # Keep words that appear at least once (you can increase min_freq to reduce vocab size)
            min_freq = 1
            words = [word for word, count in word_counts.items() if count >= min_freq]

            # Add special tokens for sequence generation
            self.itos = ["<pad>", "<start>", "<end>", "<unk>"] + sorted(words)
            self.stoi = {word: i for i, word in enumerate(self.itos)}
        else:
            self.stoi, self.itos = vocab

        print(f"--- Loaded {len(self.rows)} samples for split: '{split}' from {pairs_csv} ---")

    def _build_map(self, meta_csv):
        # ... (this method is unchanged)
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
        # ... (this method is unchanged)
        return s.strip().lower().replace(".", "").replace(",", "")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        cur_path = self.study_to_path[r["study_id"]]
        ref_path = self.study_to_path[r["ref_id"]]
        q = r["question"].strip().lower()
        a_text = self._norm(r["answer"])

        # --- NEW: Tokenize the answer sentence into a sequence of IDs ---
        tokens = ["<start>"] + a_text.split() + ["<end>"]
        answer_ids = [self.stoi.get(token, self.stoi["<unk>"]) for token in tokens]

        # Pad the sequence to a fixed length
        padded_ids = answer_ids[: self.max_ans_len]
        padded_ids += [self.stoi["<pad>"]] * (self.max_ans_len - len(padded_ids))

        single_answer_id = self.stoi.get(a_text, 0)

        img_cur = img_tf(Image.open(cur_path))
        img_ref = img_tf(Image.open(ref_path))

        return {
            "img_cur": img_cur,
            "img_ref": img_ref,
            "question": q,
            'answer_id': torch.tensor(single_answer_id, dtype=torch.long),
            "answer_ids": torch.tensor(
                padded_ids, dtype=torch.long
            ),  # Note the key is now 'answer_ids'
            "meta": (r["subject_id"], r["study_id"], r["ref_id"]),
        }
