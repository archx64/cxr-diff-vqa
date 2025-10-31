import torch
import yaml, json, argparse
from pathlib import Path
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt

# import necessary classes from your project
from lib.model import DiffVQAModel
from lib.dataset import img_tf, IMAGENET_MEAN, IMAGENET_STD
from src.train import tokenize_questions

"""
python src/inference.py \
    --config configs/clinicalbert_resnet50.yaml \
    --model_path models/drift_vqa_final.pth \
    --ref_image /path/to/your/past_image.jpg \
    --cur_image /path/to/your/current_image.jpg \
    --question "what has changed in the left lung"
"""


def deprocess_image(tensor):
    """Converts a normalized tensor back to a displayable image."""
    tensor = tensor.clone().cpu()
    for t, m, s in zip(tensor, IMAGENET_MEAN, IMAGENET_STD):
        t.mul_(s).add_(m)
    tensor = tensor.clamp(0, 1)
    return tensor.permute(1, 2, 0).numpy()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. load the training configuration
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # 2. load the vocabulary
    vocab_path = Path("models/vocab.json")
    if not vocab_path.exists():
        print(
            f"Error: Vocabulary file not found at {vocab_path}. Please run training first."
        )
        return
    with open(vocab_path, "r") as f:
        loaded_vocab = json.load(f)
    vocab = loaded_vocab["itos"]

    # 3. re-create the model architecture using config values
    model = DiffVQAModel(
        backbone=cfg.get("backbone", "resnet18"),
        text_encoder=cfg.get("text_encoder", "tiny"),
        num_classes=len(vocab),
        text_proj_dim=cfg.get("text_proj_dim", 256),
        topk=cfg.get("topk", 64),
        max_ans_len=cfg.get("max_ans_len", 16),
    ).to(device)

    # 4. load the saved model weights
    print(f"Loading trained model from: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print("Model loaded and set to evaluation mode.")

    # 5. Prepare the inputs
    img_ref_pil = Image.open(args.ref_image)
    img_cur_pil = Image.open(args.cur_image)

    img_ref = img_tf(img_ref_pil).unsqueeze(0).to(device)
    img_cur = img_tf(img_cur_pil).unsqueeze(0).to(device)

    question = [args.question.strip().lower()]
    token_batch = tokenize_questions(
        model.text, question, use_hf=getattr(model, "uses_hf", False), device=device
    )

    # 6. run inference
    with torch.no_grad():
        output = model(img_ref, img_cur, token_batch)

    # 7. interpret the output based on the model head
    predicted_answer = ""

    # pass sel_tokens to the head to generate the sequence of IDs
    _, preds_ids = model.head(output["sel_tokens"])
    preds_ids = preds_ids.cpu().tolist()[0]  # get the list of IDs for the first sample

    # convert the sequence of IDs back to a string
    pred_tokens = []
    for token_id in preds_ids:
        if token_id == 2:
            break  # Stop at <end> token
        if token_id > 2:  # Ignore <pad>, <start> tokens
            pred_tokens.append(vocab[token_id])
    predicted_answer = " ".join(pred_tokens)

    print("\n--- Inference Results ---")
    print(f"Question: {args.question}")
    print(f"Predicted Answer: {predicted_answer}")

    # 8. Visualize the heatmap
    print("\nVisualizing the model's attention heatmap...")
    heatmap = output["heatmap"].squeeze().cpu().numpy()
    img_size = img_ref_pil.size

    heatmap_resized = (
        F.interpolate(
            torch.tensor(heatmap).unsqueeze(0).unsqueeze(0),
            size=(img_size[1], img_size[0]),
            mode="bilinear",
            align_corners=False,
        )
        .squeeze()
        .numpy()
    )

    _, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img_ref_pil, cmap="gray")
    axes[0].imshow(heatmap_resized, cmap="jet", alpha=0.5)
    axes[0].set_title("Reference Image with Attention")
    axes[0].axis("off")

    axes[1].imshow(img_cur_pil, cmap="gray")
    axes[1].imshow(heatmap_resized, cmap="jet", alpha=0.5)
    axes[1].set_title("Current Image with Attention")
    axes[1].axis("off")

    plt.suptitle(f"Attention for question: '{args.question}'")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # The argparse setup for command-line use
    parser = argparse.ArgumentParser(
        description="Run inference with a trained DRIFT-VQA model."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the training YAML config file.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/drift_vqa_final.pth",
        help="Path to the trained model weights (.pth).",
    )
    parser.add_argument(
        "--ref_image",
        type=str,
        required=True,
        help="Path to the reference (past) image.",
    )
    parser.add_argument(
        "--cur_image", type=str, required=True, help="Path to the current image."
    )
    parser.add_argument(
        "--question",
        type=str,
        default="what has changed compared to the reference image?",
        help="The question to ask.",
    )
    args = parser.parse_args()
    main(args)
