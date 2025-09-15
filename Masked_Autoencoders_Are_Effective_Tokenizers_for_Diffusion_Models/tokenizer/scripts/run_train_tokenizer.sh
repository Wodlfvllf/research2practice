#!/usr/bin/env bash
#
# Script to run the first stage of MAETok pre-training.
#

# --- Configuration ---
# Path to your prepared ImageNet dataset (created by prepare_imagenet.py)
DATA_PATH="/root/PapersImplementation/Masked_Autoencoders_Are_Effective_Tokenizers_for_Diffusion_Models/data/imagenet_prepared"

# Training parameters
RESOLUTION=256
BATCH_SIZE=2
EPOCHS=1
LEARNING_RATE=1.5e-4
MASK_RATIO=0.75

# Checkpoint directory
CHECKPOINT_DIR="./tokenizer/checkpoints/maetok_pretrained"

# --- Validation ---
if [ "$DATA_PATH" == "/path/to/your/prepared/imagenet" ]; then
    echo "ERROR: Please update the DATA_PATH variable in this script to point to your prepared ImageNet directory."
    exit 1
fi

if [ ! -d "$DATA_PATH" ]; then
    echo "ERROR: The specified data path does not exist: $DATA_PATH"
    exit 1
fi

# --- Run Training ---
echo "Starting MAETok pre-training..."
echo "Dataset: $DATA_PATH"
echo "Checkpoints will be saved to: $CHECKPOINT_DIR"

# The train_tokenizer.py script is expected to be in the tokenizer/src directory.
# We can run it as a module from the project root.
python -m tokenizer.src.tokenizer.train_tokenizer \
    --data-path "$DATA_PATH" \
    --resolution "$RESOLUTION" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LEARNING_RATE" \
    --mask-ratio "$MASK_RATIO" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    # --no-aux-features  # Uncomment this if you want to disable auxiliary decoders for this run

echo "Training script finished."
