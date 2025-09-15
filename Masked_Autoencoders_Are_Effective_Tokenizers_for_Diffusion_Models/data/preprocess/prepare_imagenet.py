import os
import argparse
import shutil
from tqdm import tqdm
import requests

# The official validation ground truth labels are in XML format. A simpler, commonly used
# file maps validation image names to their class index. We'll download one if not provided.
VAL_GT_URL = "https://raw.githubusercontent.com/soumith/imagenet-torch/master/val.txt"
VAL_GT_FILENAME = "val_ground_truth.txt"

def download_val_gt(output_dir):
    """Downloads the validation ground truth file."""
    filepath = os.path.join(output_dir, VAL_GT_FILENAME)
    if os.path.exists(filepath):
        print(f"Validation ground truth file already exists at {filepath}")
        return filepath

    print(f"Downloading validation ground truth from {VAL_GT_URL}...")
    try:
        response = requests.get(VAL_GT_URL, stream=True)
        response.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded {VAL_GT_FILENAME}")
        return filepath
    except requests.exceptions.RequestException as e:
        print(f"Error downloading validation ground truth: {e}")
        print("Please ensure you have an internet connection or provide the file manually using --val-gt-path.")
        return None

def prepare_imagenet(imagenet_dir, output_dir, val_gt_path):
    """
    Prepares the ImageNet dataset by organizing it into 'train' and 'val'
    directories with class-specific subfolders.

    This script assumes the raw 'train' directory is already organized by class
    subfolders (e.g., n01440764), which is standard after extracting the tar files.
    The raw 'val' directory is assumed to contain all validation images in a flat structure.
    """
    # 1. Define paths
    raw_train_dir = os.path.join(imagenet_dir, 'train')
    raw_val_dir = os.path.join(imagenet_dir, 'val')
    
    out_train_dir = os.path.join(output_dir, 'train')
    out_val_dir = os.path.join(output_dir, 'val')

    # 2. Check if raw directories exist
    if not os.path.isdir(raw_train_dir):
        raise FileNotFoundError(f"Raw train directory not found at: {raw_train_dir}. "
                                "Please ensure it contains subdirectories for each class (e.g., n01440764).")
    if not os.path.isdir(raw_val_dir):
        raise FileNotFoundError(f"Raw validation directory not found at: {raw_val_dir}. "
                                "Please ensure it contains all validation images.")

    # 3. Create output directories
    print(f"Creating output directories at {output_dir}...")
    os.makedirs(out_train_dir, exist_ok=True)
    os.makedirs(out_val_dir, exist_ok=True)

    # 4. Process the training set
    print("Processing training set...")
    class_folders = [d for d in os.listdir(raw_train_dir) if os.path.isdir(os.path.join(raw_train_dir, d))]
    
    if not class_folders:
        raise ValueError(f"No class folders found in {raw_train_dir}. The training set might not be extracted correctly.")

    for class_folder in tqdm(class_folders, desc="Moving train classes"):
        src_path = os.path.join(raw_train_dir, class_folder)
        dest_path = os.path.join(out_train_dir, class_folder)
        if not os.path.exists(dest_path):
            shutil.move(src_path, dest_path)

    print(f"Training set processed. {len(class_folders)} classes moved.")

    # 5. Process the validation set
    print("\nProcessing validation set...")
    if not os.path.exists(val_gt_path):
        raise FileNotFoundError(f"Validation ground truth file not found at: {val_gt_path}")

    # Get the sorted list of class names (synsets) from the training directory.
    # This assumes the class indices in the ground truth file correspond to this sorted list.
    sorted_class_names = sorted(os.listdir(out_train_dir))
    if len(sorted_class_names) != 1000:
        print(f"Warning: Found {len(sorted_class_names)} classes in the training directory, but ImageNet usually has 1000.")

    with open(val_gt_path, 'r') as f:
        val_labels = [line.strip().split() for line in f.readlines()]

    if not val_labels:
        raise ValueError(f"Validation ground truth file is empty: {val_gt_path}")

    for item in tqdm(val_labels, desc="Organizing validation images"):
        if len(item) < 2:
            print(f"Skipping malformed line in ground truth file: {item}")
            continue
        img_name, class_id_str = item[0], item[1]
        
        try:
            # Ground truth is 1-indexed, so convert to 0-indexed
            class_idx = int(class_id_str) - 1
            if not (0 <= class_idx < len(sorted_class_names)):
                print(f"Warning: Class index {class_idx + 1} is out of bounds for {img_name}. Skipping.")
                continue
        except ValueError:
            print(f"Warning: Could not parse class ID '{class_id_str}' for {img_name}. Skipping.")
            continue

        class_name = sorted_class_names[class_idx]
        class_dir = os.path.join(out_val_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        src_img_path = os.path.join(raw_val_dir, img_name)
        dest_img_path = os.path.join(class_dir, img_name)
        
        if os.path.exists(src_img_path):
            shutil.move(src_img_path, dest_img_path)
        else:
            print(f"Warning: Validation image not found: {src_img_path}")

    print("Validation set processed.")
    print("\nImageNet preparation complete!")
    print(f"Formatted dataset is available at: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Prepare ImageNet dataset for PyTorch training.")
    parser.add_argument("--imagenet-dir", required=True,
                        help="Path to the raw ImageNet directory containing 'train' and 'val' folders.")
    parser.add_argument("--output-dir", required=True,
                        help="Path to the directory where the formatted dataset will be saved.")
    parser.add_argument("--val-gt-path", default=None,
                        help="Optional: Path to the validation ground truth file. "
                             f"If not provided, will attempt to download from {VAL_GT_URL}.")
    args = parser.parse_args()

    val_gt_path = args.val_gt_path
    if val_gt_path is None:
        val_gt_path = download_val_gt(args.output_dir)
        if val_gt_path is None:
            exit(1)

    try:
        prepare_imagenet(args.imagenet_dir, args.output_dir, val_gt_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        exit(1)

if __name__ == "__main__":
    main()
