#!/usr/bin/env bash
# create_maetok_repo.sh
# Creates the "PapersImplementation/Masked-Autoencoders-Are-Effective-Tokenizers-for-Diffusion-Models" repo skeleton.
# Usage:
#   chmod +x create_maetok_repo.sh
#   ./create_maetok_repo.sh

set -euo pipefail

ROOT="/root/PapersImplementation/Masked_Autoencoders_Are_Effective_Tokenizers_for_Diffusion_Models"
echo "Creating directories and files under $ROOT ..."

# list of files to create
FILES=(
  "$ROOT/README.md"
  "$ROOT/LICENSE"
  "$ROOT/CITATION.cff"

  "$ROOT/paper/2502.03444v2.pdf"
  "$ROOT/paper/README.md"

  "$ROOT/docs/architecture.md"
  "$ROOT/docs/training_recipe.md"
  "$ROOT/docs/reproduce_tables.md"

  "$ROOT/data/README.md"
  "$ROOT/data/imagenet/.gitkeep"
  "$ROOT/data/laion-coco/.gitkeep"
  "$ROOT/data/preprocess/download_scripts.sh"
  "$ROOT/data/preprocess/prepare_imagenet.py"

  "$ROOT/third_party/XQ-GAN/.gitkeep"
  "$ROOT/third_party/SiT/.gitkeep"

  "$ROOT/tokenizer/src/tokenizer/__init__.py"
  "$ROOT/tokenizer/src/tokenizer/models/vit_encoder.py"
  "$ROOT/tokenizer/src/tokenizer/models/vit_decoder.py"
  "$ROOT/tokenizer/src/tokenizer/models/aux_decoder.py"
  "$ROOT/tokenizer/src/tokenizer/models/learnable_latent_tokens.py"
  "$ROOT/tokenizer/src/tokenizer/data/imagenet_dataset.py"
  "$ROOT/tokenizer/src/tokenizer/data/transforms.py"
  "$ROOT/tokenizer/src/tokenizer/losses/pixel_loss.py"
  "$ROOT/tokenizer/src/tokenizer/losses/perceptual_loss.py"
  "$ROOT/tokenizer/src/tokenizer/losses/adversarial_loss.py"
  "$ROOT/tokenizer/src/tokenizer/utils/metrics.py"
  "$ROOT/tokenizer/src/tokenizer/utils/logger.py"
  "$ROOT/tokenizer/src/tokenizer/utils/distributed.py"
  "$ROOT/tokenizer/src/tokenizer/train_tokenizer.py"
  "$ROOT/tokenizer/src/tokenizer/finetune_decoder.py"
  "$ROOT/tokenizer/src/tokenizer/eval_tokenizer.py"
  "$ROOT/tokenizer/configs/defaults_tokenizer.py"
  "$ROOT/tokenizer/scripts/run_train_tokenizer.sh"
  "$ROOT/tokenizer/scripts/run_finetune_decoder.sh"
  "$ROOT/tokenizer/notebooks/tokenizer_quickstart.ipynb"
  "$ROOT/tokenizer/checkpoints/.gitkeep"

  "$ROOT/diffusion/src/diffusion/models/sit_model.py"
  "$ROOT/diffusion/src/diffusion/models/lightningdit_model.py"
  "$ROOT/diffusion/src/diffusion/training/train_sit.py"
  "$ROOT/diffusion/src/diffusion/training/train_lightningdit.py"
  "$ROOT/diffusion/src/diffusion/sampling/sample.py"
  "$ROOT/diffusion/src/diffusion/eval_generation.py"
  "$ROOT/diffusion/scripts/run_train_sit.sh"
  "$ROOT/diffusion/scripts/run_train_lightningdit.sh"
  "$ROOT/diffusion/checkpoints/.gitkeep"

  "$ROOT/analysis/src/gmm_analysis.py"
  "$ROOT/analysis/src/umap_viz.py"
  "$ROOT/analysis/src/linear_probe.py"
  "$ROOT/analysis/src/compute_metrics.py"
  "$ROOT/analysis/notebooks/gmm_analysis.ipynb"
  "$ROOT/analysis/notebooks/latent_viz.ipynb"

  "$ROOT/tools/feature_extractors/extract_dinov2.py"
  "$ROOT/tools/feature_extractors/extract_clip.py"
  "$ROOT/tools/feature_extractors/compute_hog.py"
  "$ROOT/tools/metrics/rfid.py"
  "$ROOT/tools/metrics/fid_wrapper.py"
  "$ROOT/tools/metrics/inception_score.py"
  "$ROOT/tools/launcher/launch_distributed.py"
  "$ROOT/tools/launcher/docker/.gitkeep"

  "$ROOT/experiments/exp_256_maetok/README.md"
  "$ROOT/experiments/exp_256_maetok/run.sh"
  "$ROOT/experiments/exp_256_maetok/results/.gitkeep"
  "$ROOT/experiments/exp_512_maetok/.gitkeep"

  "$ROOT/notebooks/demo_encode_decode.ipynb"
  "$ROOT/notebooks/demo_generation.ipynb"

  "$ROOT/tools_setup/Dockerfile"
  "$ROOT/tools_setup/environment.yml"
  "$ROOT/tools_setup/requirements.txt"

  "$ROOT/scripts/setup_env.sh"
  "$ROOT/scripts/reproduce_all_results.sh"

  "$ROOT/tests/test_models.py"
  "$ROOT/tests/test_data.py"

  "$ROOT/.github/workflows/ci.yml"

  "$ROOT/release/model_cards/MAETok_model_card.md"
  "$ROOT/release/checkpoints/.gitkeep"
)

# Create parent directories and files
for f in "${FILES[@]}"; do
  dir=$(dirname "$f")
  mkdir -p "$dir"
  # If the file is the PDF mentioned in your paper folder, do not overwrite if you already have it
  if [[ "$f" == *"2502.03444v2.pdf" ]]; then
    if [ -f "$f" ]; then
      echo "PDF already exists at $f - skipping creation"
    else
      # create an empty placeholder file (user can copy real PDF later)
      : > "$f"
    fi
  else
    # create file if not exists
    if [ ! -e "$f" ]; then
      : > "$f"
    fi
  fi
done

# Add minimal README and LICENSE and CITATION contents
cat > "$ROOT/README.md" <<'EOF'
# Masked Autoencoders Are Effective Tokenizers for Diffusion Models
Repository skeleton for reproducing the paper "Masked Autoencoders Are Effective Tokenizers for Diffusion Models".
Fill in code under the directories created by the helper script.
EOF

cat > "$ROOT/paper/README.md" <<'EOF'
Place the paper PDF and any supplemental materials here.
If you have a local copy of 2502.03444v2.pdf, copy it into this folder.
EOF

cat > "$ROOT/CITATION.cff" <<'EOF'
cff-version: 1.2.0
message: "If you use this work, please cite the original paper."
title: "Masked Autoencoders Are Effective Tokenizers for Diffusion Models"
authors:
  - family-names: "Authors"
    given-names: "Paper Authors"
year: 2025
doi: ""
EOF

cat > "$ROOT/LICENSE" <<'EOF'
MIT License
Copyright (c) YEAR AUTHOR
Permission is hereby granted...
(Replace this text with a proper license file.)
EOF

# Add minimal example content for a few key python files so they are valid python packages
cat > "$ROOT/tokenizer/src/tokenizer/__init__.py" <<'EOF'
"""
Tokenizer package for MAETok reproducibility.
"""
__all__ = []
EOF

cat > "$ROOT/tokenizer/src/tokenizer/train_tokenizer.py" <<'EOF'
#!/usr/bin/env python3
# Minimal argparse skeleton for tokenizer training.
import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-path', required=False, default='')
    p.add_argument('--resolution', type=int, default=256)
    return p.parse_args()

def main():
    args = parse_args()
    print("This is a placeholder train_tokenizer script. Fill with training loop.")

if __name__ == '__main__':
    main()
EOF
chmod +x "$ROOT/tokenizer/src/tokenizer/train_tokenizer.py"

# Helpful note for user: copy your uploaded PDF into the paper folder if available
echo
echo "Structure created under: $ROOT"
echo
echo "If you have the paper PDF locally, copy it into the paper folder with e.g.:"
echo "  cp /path/to/2502.03444v2.pdf \"$ROOT/paper/2502.03444v2.pdf\""
echo
echo "Done."

