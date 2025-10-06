import torch
import argparse
import yaml
from PIL import Image

# Import necessary classes from your project
from lib.models import DiffVQAModel
from lib.dataset import img_tf  # Use the same image transforms

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load the training configuration
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # 2. Re-create the model architecture with the same config
    # We need the vocab from the training set to map output IDs to answers
    # This is a bit of a shortcut; ideally, you'd save the vocab with the model
    from lib.dataset import DiffVQADataset
    train_ds = DiffVQADataset(cfg['data_root'], cfg['pairs_csv'], cfg['meta_csv'], split="train")
    vocab = train_ds.itos # list of answer strings, index is the class id

    model = DiffVQAModel(
        backbone=cfg.get('backbone', 'resnet50'),
        text_encoder=cfg.get('text_encoder', 'tiny'),
        text_proj_dim=cfg.get('text_proj_dim', 256),
        topk=cfg.get('topk', 64),
        num_classes=len(vocab),
        head=cfg.get('head', 'classifier')
    ).to(device)

    # 3. Load the saved model weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print("Model loaded and set to evaluation mode.")

    # 4. Prepare the inputs
    # Load and transform images
    img_ref = img_tf(Image.open(args.ref_image)).unsqueeze(0).to(device)
    img_cur = img_tf(Image.open(args.cur_image)).unsqueeze(0).to(device)
    
    # Prepare the question
    question = [args.question.strip().lower()]
    
    # Tokenize the question using the model's text encoder
    from train import tokenize_questions # We can reuse this helper
    token_batch = tokenize_questions(
        model.text, question, use_hf=getattr(model, "uses_hf", False), device=device
    )

    # 5. Run inference
    with torch.no_grad():
        output = model(img_ref, img_cur, token_batch)

    # 6. Interpret the output
    if model.is_classifier:
        logits = output['logits']
        predicted_id = logits.argmax(-1).item()
        predicted_answer = vocab[predicted_id]
        print("\n--- Inference Results ---")
        print(f"Question: {args.question}")
        print(f"Predicted Answer: {predicted_answer}")
    else:
        # Inference for a decoder model
        _, predicted_ids = model.head(output['sel_tokens'])
        # You would need a tokenizer to convert these IDs back to words
        print("\nDecoder output IDs:", predicted_ids.cpu().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a trained DRIFT-VQA model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the training YAML config file.")
    parser.add_argument("--model_path", type=str, default="drift_vqa_final.pth", help="Path to the trained model weights (.pth).")
    parser.add_argument("--ref_image", type=str, required=True, help="Path to the reference (past) image.")
    parser.add_argument("--cur_image", type=str, required=True, help="Path to the current image.")
    parser.add_argument("--question", type=str, default="what has changed compared to the reference image?", help="The question to ask.")
    
    args = parser.parse_args()
    main(args)