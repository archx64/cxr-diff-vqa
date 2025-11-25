import logging, colorlog, json
from logging.handlers import RotatingFileHandler
from torch.utils.data import DataLoader
import torch

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider


RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
COLOR_END = "\033[0m"


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


def collate(batch):
    # simple default collator (images are same size from transforms)
    out = {k: [] for k in batch[0].keys()}
    for b in batch:
        for k, v in b.items():
            out[k].append(v)
    out["img_cur"] = torch.stack(out["img_cur"])
    out["img_ref"] = torch.stack(out["img_ref"])

    if "answer_id" in out:
        out["answer_id"] = torch.stack(out["answer_id"])

    if "answer_ids" in out:
        out["answer_ids"] = torch.stack(out["answer_ids"])

    return out


def make_loader(ds, bs, shuffle, num_workers=4):
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate,
    )


def setup_logging(log_file: str, console_level=logging.DEBUG):
    """
    Sets up the root logger to log to a file and the console.
    log_file: The file to log to.
    console_level: The minimum level to show in the console (e.g., logging.WARNING)
    """
    logger = logging.getLogger() # get the root logger
    logger.setLevel(logging.DEBUG) # set the lowest level to process (DEBUG)

    # clear any existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.propagate = True

    # File Handler
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10485760, # 10MB
        backupCount=5,
        mode="a",
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG) # Log everything to the file
    logger.addHandler(file_handler)

    # Console Handler (respects the new 'console_level')
    console_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        log_colors={
            "DEBUG": "white",
            "INFO": "cyan",
            "SUCCESS": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_white, bg_red",
        },
    )
    console_handler = colorlog.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(console_level) # Set the desired console log level
    logger.addHandler(console_handler)

    return logger


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
        print("pre-processing gts and res for scorers...")
        gts_orig = {
            img_id: [ann["caption"] for ann in anns]
            for img_id, anns in gts_orig.items()
        }
        res_orig = {
            img_id: [ann["caption"] for ann in anns]
            for img_id, anns in res_orig.items()
        }

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
        # compute scores
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
