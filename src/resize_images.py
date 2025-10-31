import argparse
from pathlib import Path
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from lib.utils import setup_logging
import os

log_file ="logs/resize.log"
logger = setup_logging(log_file=log_file)

def resize_image(args):
    """
    Worker function to resize a single image.
    Creates parent directories if they don't exist.
    """
    source_path, dest_path, size = args
    try:
        # ensure the destination directory exists
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # open, resize, and save the image
        with Image.open(source_path) as img:

            # using LANCZOS for high-quality downsampling
            img_resized = img.resize((size, size), Image.Resampling.LANCZOS)

            # convert to RGB if it's grayscale to ensure consistency
            if img_resized.mode != 'RGB':
                img_resized = img_resized.convert('RGB')
            img_resized.save(dest_path, 'JPEG', quality=95)

        return None
    except Exception as e:
        logger.error(f"Error processing {source_path}: {e}")
        return f"Error processing {source_path}: {e}"

def main():
    parser = argparse.ArgumentParser(description="Resize a directory of images while preserving the folder structure.")
    parser.add_argument("--source", type=str, required=True, help="Path to the source directory with original images.")
    parser.add_argument("--dest", type=str, required=True, help="Path to the destination directory for resized images.")
    parser.add_argument("--size", type=int, default=224, help="The target size (e.g., 224 for 224x224).")
    args = parser.parse_args()

    source_dir = Path(args.source)
    dest_dir = Path(args.dest)
    size = args.size

    logger.info(f"Source directory: {source_dir}")
    logger.info(f"Destination directory: {dest_dir}")
    logger.info(f"Target size: {size}x{size}")

    # 1. recursively find all .jpg files in the source directory
    logger.info("Finding all JPG files...")
    source_files = list(source_dir.rglob('*.jpg'))
    if not source_files:
        logger.error("Error: No .jpg files found in the source directory.")
        return
    logger.info(f"Found {len(source_files)} images to resize.")

    # 2. prepare the list of tasks for the process pool
    tasks = []
    for source_path in source_files:
        # Determine the relative path to preserve the structure
        relative_path = source_path.relative_to(source_dir)
        # Create the full destination path
        dest_path = dest_dir / relative_path
        tasks.append((source_path, dest_path, size))

    # 3. use a process pool to resize images in parallel
    logger.info("Starting image resizing (this may take a long time)...")
    
    # use os.cpu_count() to leverage all available CPU cores
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(executor.map(resize_image, tasks), total=len(tasks)))

    # report any errors that occurred
    errors = [res for res in results if res is not None]
    if errors:
        logger.info("\n--- Errors occurred during processing ---")
        for error in errors:
            logger.info(error)
    
    logger.info("\nResizing complete!")
    logger.info(f"Resized images are saved in: {dest_dir}")

if __name__ == "__main__":
    main()