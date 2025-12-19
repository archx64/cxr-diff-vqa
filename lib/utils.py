import logging, colorlog, json, nltk, os
import torch
import numpy as np

from radgraph import F1RadGraph
from logging.handlers import RotatingFileHandler
from torch.utils.data import DataLoader

from pycocotools.coco import COCO

from pycocoevalcap.eval import COCOEvalCap
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
# from pycocoevalcap.meteor.meteor import Meteor

from nltk.translate.meteor_score import meteor_score

NEPTUNE_PROJECT = "DRIFT/medical-diff-vqa"
NEPTUNE_API_TOKEN = ""

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"

CYAN = "\033[96m"
COLOR_END = "\033[0m"


def run_nlg_evaluation(gts, res):
    """
    Calculates standard NLG metrics AND RadGraph F1.
    Returns a dictionary of scores.
    """
    # 1. Setup standard COCO eval

    if os.path.exists('gts.json'): 
        os.remove('gts.json')
    if os.path.exists('res.json'): 
        os.remove('res.json')

    with open('gts.json', 'w') as f: json.dump(gts, f)
    with open('res.json', 'w') as f: json.dump(res, f)

    coco = COCO('gts.json')
    coco_result = coco.loadRes('res.json')
    coco_eval = CustomCOCOEvalCap(coco, coco_result)
    
    # Suppress standard output during evaluation to keep logs clean
    coco_eval.evaluate()
    
    # Create a dictionary to hold all results
    final_scores = {}
    for metric, score in coco_eval.eval.items():
        final_scores[metric] = score

    # 2. Calculate RadGraph F1 (The new part)
    print("Calculating RadGraph F1 (this may take a moment)...")
    try:
        # Prepare lists for RadGraph
        refs = []
        preds = []
        
        # Sort by image_id to ensure alignment
        img_ids = sorted(coco.imgs.keys())
        
        for img_id in img_ids:
            # Get ground truth
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            refs.append(anns[0]['caption'])
            
            # Get prediction
            pred_ann = coco_result.imgToAnns[img_id][0]
            preds.append(pred_ann['caption'])

        # --- SANITIZATION STEP (FIX FOR CRASH) ---
        # Replace empty or very short predictions (like "no", "yes") 
        # with a placeholder to prevent RadGraph from crashing.
        clean_preds = []
        for p in preds:
            # If prediction is empty or less than 3 chars (e.g. "no")
            if not p or len(p.strip()) < 3:
                clean_preds.append("there are no findings") 
            else:
                clean_preds.append(p)
                

        # Initialize RadGraph evaluator
        radgraph_evaluator = F1RadGraph(reward_level="all")
        
        # The evaluator returns 4 values: mean_reward, reward_list, hyp_annots, ref_annots
        # mean_reward is a tuple: (Entity F1, ER F1, Scaled F1)
        output = radgraph_evaluator(refs=refs, hyps=clean_preds)
        
        mean_rewards = output[0] # (rg_e, rg_er, rg_bar_er)
        rg_er_mean = mean_rewards[1] # Use Entity+Relation (Standard RadGraph F1)
        
        final_scores["RadGraph_F1"] = rg_er_mean
        
    except Exception as e:
        # Catch any other errors so training doesn't stop
        print(f"RadGraph calculation failed: {e}")
        final_scores["RadGraph_F1"] = 0.0
        

    # Print results to console
    print("\n--- Validation Scores ---")
    for k, v in final_scores.items():
        print(f"{k}: {v:.4f}")

    return final_scores


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
        pin_memory=False,
        collate_fn=collate,
    )

class PackageLogFilter(logging.Filter):
    """
    A custom filter to suppress DEBUG logs from specific noisy libraries
    while keeping their WARNING/ERROR logs and allowing DEBUG logs from 
    our own model code.
    """
    def filter(self, record):
        # List of packages to silence DEBUG logs for
        silenced_packages = [
            "urllib3", 
            "huggingface_hub", 
            "timm", 
            "neptune", 
            "bravado", 
            "bravado_core", 
            "swagger_spec_validator",
            "fsspec",
            "transformers"
        ]
        
        # If the log is DEBUG level (or lower)
        if record.levelno <= logging.DEBUG:
            # Check if it comes from a silenced package
            for pkg in silenced_packages:
                if record.name.startswith(pkg):
                    return False # Reject this log record
        
        return True # Allow all other logs (including our own DEBUG logs)


def setup_logging(log_file: str, console_level=logging.DEBUG):
    """
    Sets up the root logger to log to a file and the console.
    log_file: The file to log to.
    console_level: The minimum level to show in the console (e.g., logging.WARNING)
    """
    logger = logging.getLogger()  # get the root logger
    logger.setLevel(logging.DEBUG)  # set the lowest level to process (DEBUG)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("timm").setLevel(logging.WARNING)
    logging.getLogger("neptune").setLevel(logging.WARNING)
    
    # --- ADDED: Silence Neptune/Bravado dependencies ---
    logging.getLogger("bravado").setLevel(logging.WARNING)
    logging.getLogger("bravado_core").setLevel(logging.WARNING)
    logging.getLogger("swagger_spec_validator").setLevel(logging.WARNING)

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
        maxBytes=10485760,  # 10MB
        backupCount=5,
        mode="a",
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)  # Log everything to the file
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
    console_handler.setLevel(console_level)  # Set the desired console log level
    logger.addHandler(console_handler)

    return logger


class MeteorNLTK:
    def __init__(self):
        try:
            nltk.data.find('corpora/wordnet.zip')
        except LookupError:
            nltk.download('wordnet')
            nltk.download('omw-1.4')

    def compute_score(self, gts, res):
        scores = []
        for img_id in gts:
            # gts[img_id] is a list of reference strings
            # res[img_id] is a list containing the hypothesis string
            
            # NLTK expects lists of tokens (words), so we split() the strings
            references = [r.split() for r in gts[img_id]]
            hypothesis = res[img_id][0].split()
            
            # Calculate METEOR score for this single sample
            score = meteor_score(references, hypothesis)
            scores.append(score)

        # Return average score and list of individual scores
        return np.mean(scores), scores

    def method(self):
        return "METEOR"


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
            (MeteorNLTK(), "METEOR"),
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
