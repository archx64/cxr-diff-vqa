import yaml, json
import torch
from pathlib import Path

# necessary components from DRIFT
from lib.dataset import DiffVQADataset
from lib.utils import make_loader, setup_logging, run_nlg_evaluation
from lib.model import DiffVQAModel
from src.train import tokenize_questions


def main(args):
    logger = setup_logging(log_file="logs/test.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Loaded configuration from {args.config}")

    vocab_path = Path("models/vocab.json")
    if not vocab_path.exists():
        logger.error(
            f"Vocabulary file not found at {vocab_path}. Please run training first to generate it."
        )
        return

    logger.info(f"Loading vocabulary from {vocab_path}...")
    with open(vocab_path, "r") as f:
        loaded_vocab = json.load(f)
    vocab = (loaded_vocab["stoi"], loaded_vocab["itos"])

    num_classes = len(vocab[1])
    logger.info(f"Vocabulary loaded. Number of answer classes: {num_classes}")

    logger.info("Loading the test dataset...")
    test_ds = DiffVQADataset(
        cfg["data_root"],
        cfg["test_pairs_csv"],
        cfg["test_meta_csv"],
        split="test",
        vocab=vocab,
    )
    test_loader = make_loader(test_ds, bs=cfg.get("bs", 1), shuffle=False)
    logger.info(f"Test dataset loaded with {len(test_ds)} samples.")

    model = DiffVQAModel(
        backbone=cfg.get("backbone"),
        text_encoder=cfg.get("text_encoder"),
        text_model_name=cfg.get("text_model_name"),
        text_dim=cfg.get("text_dim"),
        text_finetune=cfg.get('text_finetune'),
        num_classes=num_classes,
        text_proj_dim=cfg.get("text_proj_dim"),
        topk=cfg.get("topk", 64),
        max_ans_len=cfg.get("max_ans_len"),
    ).to(device)

    logger.info(f"Loading trained model weights from {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Dictionaries to store results in COCO format
    gts = {"info": [], "images": [], "annotations": []}
    res = []

    # Use a counter for unique sample IDs
    sample_id_counter = 0

    with torch.no_grad():  # Disable gradient calculations for inference
        for batch in test_loader:
            img_cur = batch["img_cur"].to(device)
            img_ref = batch["img_ref"].to(device)
            qs = batch["question"]

            ground_truth_ids = batch["answer_ids"]  # Sequence of IDs

            tokens = tokenize_questions(
                model.text, qs, device=device
            )
            out = model(img_ref, img_cur, tokens)

            _, preds_ids = model.head(out["sel_tokens"])
            preds_ids = preds_ids.cpu().tolist()

            # Decoder ground truth is already a sequence
            ground_truth_ids_cpu = ground_truth_ids.cpu()

            for i in range(len(qs)):
                # Convert ground truth sequence to a clean string
                gt_tokens = [
                    vocab[1][token_id]
                    for token_id in ground_truth_ids_cpu[i].tolist()
                    if token_id > 2  # >2 skips pad, start, end
                ]
                gt_answer_string = " ".join(gt_tokens)

                # Convert predicted IDs to a clean string
                pred_tokens = []
                for token_id in preds_ids[i]:
                    if token_id == 2:
                        break  # Stop at <end> token
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
                    {"image_id": sample_id_counter, "id": sample_id_counter, "caption": pred_answer_string}
                )
                sample_id_counter += 1

    # Run the final NLG evaluation
    if res:
        run_nlg_evaluation(gts, res)
    else:
        logger.warning("No results were generated to evaluate.")
