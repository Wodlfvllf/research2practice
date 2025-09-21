# Masked Autoencoders Are Effective Tokenizers for Diffusion Models

This repository provides an implementation of the paper "Masked Autoencoders Are Effective Tokenizers for Diffusion Models".

## Introduction

This project explores the use of Masked Autoencoders (MAE) as tokenizers for Diffusion Models. The core idea is to leverage the powerful representation learning of MAEs to create a more effective and efficient tokenization process for training diffusion-based generative models.

## Repository Structure

```
.
├── analysis/         # Scripts for analysis and visualization
├── data/             # Data preparation scripts and storage
├── diffusion/        # Diffusion model implementation
├── docs/             # Documentation
├── experiments/      # Experiment configurations and scripts
├── notebooks/        # Jupyter notebooks for demos
├── scripts/          # Main scripts for setup and reproduction
├── tests/            # Tests for the implementation
├── tokenizer/        # MAE Tokenizer implementation
└── tools/            # Various tools and utilities
```

## Setup

1.  **Create the environment:**
    ```bash
    sh scripts/setup_env.sh
    ```
    This will set up the necessary environment using Conda and install the required packages from `tools_setup/environment.yml` and `tools_setup/requirements.txt`.

2.  **Activate the environment:**
    ```bash
    conda activate mae_tok
    ```

## Data Preparation

The models are trained on the ImageNet dataset.

1.  **Download ImageNet:** Use the scripts in `data/preprocess/` to download and prepare the Imagenet dataset.
2.  **Create Dummy Data:** For testing purposes, you can create dummy data by running:
    ```bash
    sh create.sh
    ```

## Training

The training process involves two main stages: training the MAE tokenizer and training the diffusion model.

1.  **Train the MAE Tokenizer:**
    ```bash
    sh tokenizer/scripts/run_train_tokenizer.sh
    ```

2.  **Train the Diffusion Model:**
    ```bash
    sh diffusion/scripts/run_train_lightningdit.sh
    ```

## Evaluation

To evaluate the models, use the scripts provided in the `tokenizer/src/tokenizer/` and `diffusion/src/diffusion/` directories.

-   **Tokenizer Evaluation:** `python tokenizer/src/tokenizer/eval_tokenizer.py`
-   **Diffusion Model Evaluation:** `python diffusion/src/diffusion/eval_generation.py`

## Pre-trained Models

Pre-trained model checkpoints can be found in the `release/checkpoints/` and `tokenizer/checkpoints/` directories.

## Citation

If you use this work, please cite the original paper:

```
@misc{authors2025masked,
      title={Masked Autoencoders Are Effective Tokenizers for Diffusion Models}, 
      author={Paper Authors},
      year={2025},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```