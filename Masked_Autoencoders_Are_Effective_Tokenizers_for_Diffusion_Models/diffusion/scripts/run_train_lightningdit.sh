#!/usr/bin/env bash
#
# Script to run the training for the LightningDiT diffusion model.
#

# --- Configuration ---
# Path to your prepared ImageNet dataset
DATA_PATH="/root/PapersImplementation/Masked_Autoencoders_Are_Effective_Tokenizers_for_Diffusion_Models/data/imagenet_prepared"

# Path to your fine-tuned MAETok checkpoint
TOKENIZER_PATH="./tokenizer/checkpoints/maetok_finetuned/finetuned_decoder_epoch_0.pth"

# Training parameters
BATCH_SIZE=2
EPOCHS=1
LEARNING_RATE=1e-4

# Checkpoint directory
CHECKPOINT_DIR="./diffusion/checkpoints/lightningdit_model"

# --- Validation ---
if [ "$DATA_PATH" == "/path/to/your/prepared/imagenet" ]; then
    echo "ERROR: Please update the DATA_PATH variable in this script."
    exit 1
fi

if [ ! -f "$TOKENIZER_PATH" ]; then
    echo "ERROR: MAETok checkpoint not found at: $TOKENIZER_PATH"
    exit 1
fi

# --- Run Training ---
echo "Starting LightningDiT model training..."
echo "Dataset: $DATA_PATH"
echo "Tokenizer: $TOKENIZER_PATH"
echo "Checkpoints will be saved to: $CHECKPOINT_DIR"

python -m diffusion.src.diffusion.training.train_lightningdit \
    --data-path "$DATA_PATH" \
    --tokenizer-path "$TOKENIZER_PATH" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LEARNING_RATE" \
    --checkpoint-dir "$CHECKPOINT_DIR"

echo "Training script finished."
