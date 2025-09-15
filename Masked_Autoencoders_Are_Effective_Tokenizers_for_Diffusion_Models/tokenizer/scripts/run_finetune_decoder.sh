#!/usr/bin/env bash
#
# Script to run the second stage of MAETok training: fine-tuning the decoder.
#

# --- Configuration ---
# Path to your prepared ImageNet dataset
DATA_PATH="/root/PapersImplementation/Masked_Autoencoders_Are_Effective_Tokenizers_for_Diffusion_Models/data/imagenet_prepared"

# Path to the checkpoint from the first stage of pre-training
PRETRAINED_CHECKPOINT="./tokenizer/checkpoints/maetok_pretrained/mae_best_model.pth"

# Fine-tuning parameters
BATCH_SIZE=2
EPOCHS=1
LEARNING_RATE=5e-5  # A smaller LR is common for fine-tuning
MASK_RATIO=0.75

# Checkpoint directory for the fine-tuned model
CHECKPOINT_DIR="./tokenizer/checkpoints/maetok_finetuned"

# --- Validation ---
if [ "$DATA_PATH" == "/path/to/your/prepared/imagenet" ]; then
    echo "ERROR: Please update the DATA_PATH variable in this script."
    exit 1
fi

if [ ! -f "$PRETRAINED_CHECKPOINT" ]; then
    echo "ERROR: Pre-trained checkpoint not found at: $PRETRAINED_CHECKPOINT"
    echo "Please run the first stage of training (run_train_tokenizer.sh) first."
    exit 1
fi

# --- Run Fine-tuning ---
echo "Starting MAETok decoder fine-tuning..."
echo "Dataset: $DATA_PATH"
echo "Loading pre-trained model from: $PRETRAINED_CHECKPOINT"
echo "Fine-tuned checkpoints will be saved to: $CHECKPOINT_DIR"

python -m tokenizer.src.tokenizer.finetune_decoder \
    --data-path "$DATA_PATH" \
    --pretrained-checkpoint "$PRETRAINED_CHECKPOINT" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LEARNING_RATE" \
    --mask-ratio "$MASK_RATIO" \
    --checkpoint-dir "$CHECKPOINT_DIR"

echo "Fine-tuning script finished."
