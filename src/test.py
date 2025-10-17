import yaml
import torch
import json

# --- NEW IMPORTS ---
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

# --------------------

# Import necessary components from your project
from lib.dataset import DiffVQADataset
from lib.utils import make_loader, setup_logging
from src.train import DiffVQAModel
from src.train import tokenize_questions


class CustomCOCOEvalCap(COCOEvalCap):
    def evaluate(self):
        """
        This is a copy of the original evaluate method, with the SPICE scorer removed.
        """
        imgIds = self.params["image_id"]
        gts_orig = self.coco.imgToAnns
        res_orig = self.cocoRes.imgToAnns

        # =================================================
        # preprocess ground truths and results to the format scorers expect
        # scorers expect a dict mapping image_id to a list of strings
        # =================================================
        print('pre-processing gts and res for scorers...')
        gts_orig = {img_id: [ann['caption'] for ann in anns] for img_id, anns in gts_orig.items()}
        res_orig = {img_id: [ann['caption'] for ann in anns] for img_id, anns in res_orig.items()}

        # =================================================
        # set up scorers
        # =================================================
        print("setting up scorers...")
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            # SPICE scorer is now completely removed from this list.
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print("computing %s score..." % (scorer.method()))
            score, scores = scorer.compute_score(gts_orig, res_orig)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts_orig.keys(), m)
                    print("%s: %0.3f" % (m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts_orig.keys(), method)
                print("%s: %0.3f" % (method, score))
        self.setEvalImgs()


def run_nlg_evaluation(gts, res):
    """
    Takes ground truth and result dictionaries in COCO format,
    runs the pycocoevalcap evaluator, and prints the results.
    """
    # create a temporary JSON file for ground truths
    with open("gts.json", "w") as f:
        json.dump(gts, f)

    # create a temporary JSON file for results
    with open("res.json", "w") as f:
        json.dump(res, f)

    coco = COCO("gts.json")
    coco_result = coco.loadRes("res.json")

    coco_eval = CustomCOCOEvalCap(coco, coco_result)
    coco_eval.evaluate()

    print("\n--- NLG Evaluation Metrics ---")
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")


def main(args):
    logger = setup_logging(log_file="logs/test.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Loaded configuration from {args.config}")

    logger.info("Rebuilding vocabulary from the training set...")
    train_ds = DiffVQADataset(
        cfg["data_root"], cfg["train_pairs_csv"], cfg["train_meta_csv"], split="train"
    )
    vocab = (train_ds.stoi, train_ds.itos)
    num_classes = len(vocab[1])
    logger.info(f"Vocabulary built. Number of answer classes: {num_classes}")

    logger.info("Loading the test dataset...")
    test_ds = DiffVQADataset(
        cfg["data_root"],
        cfg["test_pairs_csv"],
        cfg["test_meta_csv"],
        split="test",
        vocab=vocab,
    )
    test_loader = make_loader(test_ds, bs=cfg.get("bs", 8), shuffle=False)
    logger.info(f"Test dataset loaded with {len(test_ds)} samples.")

    model = DiffVQAModel(
        backbone=cfg.get("backbone", "resnet50"),
        text_encoder=cfg.get("text_encoder", "tiny"),
        num_classes=num_classes,
        head=cfg.get("head", "classifier"),
        text_proj_dim=cfg.get("text_proj_dim", 256),
        topk=cfg.get("topk", 64),
    ).to(device)

    logger.info(f"Loading trained model weights from {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # --- MODIFIED EVALUATION LOOP ---
    # Dictionaries to store results in COCO format
    gts = {"info": [], "images": [], "annotations": []}
    res = []

    # Use a counter for unique sample IDs
    sample_id_counter = 0

    with torch.no_grad():
        for batch in test_loader:
            # ... (loading data to device is the same)
            img_cur = batch["img_cur"].to(device)
            img_ref = batch["img_ref"].to(device)
            answers = batch["answer_id"]
            qs = batch["question"]

            tokens = tokenize_questions(
                model.text, qs, use_hf=getattr(model, "uses_hf", False), device=device
            )
            out = model(img_ref, img_cur, tokens)

            if model.is_classifier:
                preds_ids = out["logits"].argmax(dim=-1).cpu()

                for i in range(len(qs)):
                    gt_answer_string = vocab[1][answers[i].item()]
                    pred_answer_string = vocab[1][preds_ids[i].item()]

                    # --- ADDED/MODIFIED LINES ---
                    # 1. Add an entry to the 'images' list for each sample
                    gts["images"].append({"id": sample_id_counter})

                    # 2. Add the required unique 'id' key to each annotation
                    gts["annotations"].append(
                        {
                            "image_id": sample_id_counter,
                            "id": sample_id_counter,  # <-- THE FIX: Add a unique annotation ID
                            "caption": gt_answer_string,
                        }
                    )
                    # --- END OF CHANGES ---

                    res.append(
                        {"image_id": sample_id_counter, "caption": pred_answer_string}
                    )

                    sample_id_counter += 1
            else:
                logger.warning(
                    "NLG metric evaluation for decoder head is not implemented yet."
                )

    if res:
        run_nlg_evaluation(gts, res)
    else:
        logger.warning("No results were generated to evaluate.")
