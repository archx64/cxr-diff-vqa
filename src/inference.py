import torch
import argparse
import yaml
import json
from pathlib import Path
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np # Added for heatmap processing
import logging

# Import project components
from lib.model import DiffVQAModel
from lib.dataset import IMAGENET_MEAN, IMAGENET_STD, gray_to_rgb
from src.train import tokenize_questions
from lib.utils import setup_logging # For logging

# Setup logger
logger = logging.getLogger(__name__)

# Local inference transform (includes resize)
inference_img_tf = transforms.Compose([
    transforms.Resize((224, 224)), # Resize to model's expected input
    transforms.Lambda(gray_to_rgb),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

def create_heatmap(feature_map, img_size):
    """
    Takes a [C, H, W] feature map, aggregates it to [H, W],
    and upscales to the original image size.
    """
    try:
        # 1. Aggregate across the channel dimension: [C, H, W] -> [H, W]
        # We use .abs() to capture magnitude, then mean
        heatmap = torch.abs(feature_map).mean(dim=0).cpu().numpy()
        
        # 2. Normalize the heatmap to [0, 1] for visualization
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-6)
        
        # 3. Upscale the heatmap to the original image size
        heatmap_resized = F.interpolate(
            torch.tensor(heatmap).unsqueeze(0).unsqueeze(0),
            size=(img_size[1], img_size[0]), # (Height, Width)
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()
        
        return heatmap_resized
    except Exception as e:
        logger.error(f"Error creating heatmap: {e}")
        # Return a blank map on error
        return np.zeros((img_size[1], img_size[0]))

def main(args):
    # Setup logger for inference
    global logger
    if not logger.hasHandlers():
        logger = setup_logging(log_file="logs/inference.log", console_level=logging.INFO)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 1. Load the training configuration
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Loaded configuration from {args.config}")

    # 2. Load the vocabulary
    vocab_path = Path("models/vocab.json")
    if not vocab_path.exists():
        logger.error(f"Vocabulary file not found at {vocab_path}. Please run training first.")
        return
    with open(vocab_path, 'r') as f:
        loaded_vocab = json.load(f)
    vocab = loaded_vocab['itos']
    num_classes = len(vocab)
    logger.info(f"Vocabulary loaded. Size: {num_classes}")

    # 3. Re-create the model architecture (Decoder-Only)
    model = DiffVQAModel(
        backbone=cfg.get('backbone', 'resnet18'),
        text_encoder=cfg.get('text_encoder'),
        text_model_name=cfg.get('text_model_name'),
        text_dim=cfg.get('text_dim'),
        text_proj_dim=cfg.get('text_proj_dim'),
        text_finetune=cfg.get('text_finetune', False),
        topk=cfg.get('topk', 64),
        num_classes=num_classes,
        max_ans_len=cfg.get('max_ans_len', 48)
    ).to(device)

    # 4. Load the saved model weights
    logger.info(f"Loading trained model from: {args.model_path}")
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        logger.info("Model loaded and set to evaluation mode.")
    except Exception as e:
        logger.error(f"Error loading model weights: {e}")
        logger.error("This is often due to a mismatch between your config file and the saved model.")
        return

    # 5. Prepare the inputs
    try:
        img_ref_pil = Image.open(args.ref_image)
        img_cur_pil = Image.open(args.cur_image)
    except Exception as e:
        logger.error(f"Error opening image files: {e}")
        return

    img_ref = inference_img_tf(img_ref_pil).unsqueeze(0).to(device)
    img_cur = inference_img_tf(img_cur_pil).unsqueeze(0).to(device)

    question = [args.question.strip().lower()]
    token_batch = tokenize_questions(
        model.text, question, device=device
    )

    # 6. Run inference
    with torch.no_grad():
        output = model(img_ref, img_cur, token_batch)

    # 7. Generate Answer (Decoder-Only)
    _, preds_ids = model.head(output['sel_tokens'])
    preds_ids = preds_ids.cpu().tolist()[0]
    pred_tokens = []
    for token_id in preds_ids:
        if token_id == 2: break # Stop at <end> token
        if token_id > 2: # Ignore <pad> and <start>
            pred_tokens.append(vocab[token_id])
    predicted_answer = " ".join(pred_tokens)

    print("\n--- Inference Results ---")
    print(f"Question: {args.question}")
    print(f"Predicted Answer: {predicted_answer}")

    # 8. Visualize the TWO new Heatmaps
    print("\nVisualizing directional heatmaps...")
    
    r_neg_map = output['r_neg'].squeeze(0) # Features that are "Gone"
    r_pos_map = output['r_pos'].squeeze(0) # Features that are "New"
    img_size = img_ref_pil.size 

    heatmap_gone = create_heatmap(r_neg_map, img_size)
    heatmap_new = create_heatmap(r_pos_map, img_size)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot "Gone" Heatmap (R-) on Reference Image
    axes[0].imshow(img_ref_pil, cmap='gray')
    axes[0].imshow(heatmap_gone, cmap='jet', alpha=0.5)
    axes[0].set_title('Reference Image (Features that "Gone" [R-])')
    axes[0].axis('off')

    # Plot "New" Heatmap (R+) on Current Image
    axes[1].imshow(img_cur_pil, cmap='gray')
    axes[1].imshow(heatmap_new, cmap='jet', alpha=0.5)
    axes[1].set_title('Current Image (Features that are "New" [R+])')
    axes[1].axis('off')

    plt.suptitle(f"Q: '{args.question}' \n A: '{predicted_answer}\n'")
    # plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Setup logger for terminal use
    Path("logs").mkdir(exist_ok=True)
    logger = setup_logging(log_file="logs/inference.log", console_level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run inference with a trained DRIFT-VQA model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the training YAML config file.")
    parser.add.argument("--model_path", type=str, default="models/drift_vqa_final.pth", help="Path to the trained model weights (.pth).")
    parser.add_argument("--ref_image", type=str, required=True, help="Path to the reference (past) image.")
    parser.add_argument("--cur_image", type=str, required=True, help="Path to the current image.")
    parser.add_argument("--question", type=str, default="what has changed compared to the reference image?", help="The question to ask.")
    args = parser.parse_args()
    main(args)