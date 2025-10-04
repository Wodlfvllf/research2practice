# Super Weight Analysis Implementation Guide

## Project Overview

This project implements the super weight detection and analysis method from "The Super Weight in Large Language Models" paper. We'll analyze Qwen2/Qwen3 models from Hugging Face to identify and study their super weights.

## Directory Structure

```
superweight-analysis/
├── README.md
├── requirements.txt
├── config/
│   └── model_config.yaml
├── src/
│   ├── __init__.py
│   ├── model_loader.py          # Load HuggingFace models
│   ├── super_weight_detector.py  # Core detection algorithm
│   ├── activation_analyzer.py    # Analyze activations
│   ├── visualization.py          # Plotting and visualization
│   └── utils.py                  # Helper functions
├── experiments/
│   ├── __init__.py
│   ├── detect_super_weights.py   # Main detection script
│   ├── validate_super_weights.py # Validation experiments
│   └── quantization_tests.py     # Quantization with super weights
├── notebooks/
│   ├── 01_exploration.ipynb      # Initial exploration
│   ├── 02_detection.ipynb        # Detection walkthrough
│   └── 03_analysis.ipynb         # Results analysis
├── results/
│   ├── super_weight_coordinates/
│   ├── visualizations/
│   └── metrics/
└── tests/
    ├── __init__.py
    ├── test_detector.py
    └── test_activation_analyzer.py
```

---

## Implementation Plan

### Phase 1: Environment Setup

#### 1.1 Dependencies (`requirements.txt`)
```txt
torch>=2.0.0
transformers>=4.35.0
accelerate>=0.24.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
pyyaml>=6.0
tqdm>=4.65.0
scipy>=1.11.0
jupyter>=1.0.0
lm-eval>=0.4.0  # For evaluation harness
```

#### 1.2 Configuration File (`config/model_config.yaml`)
```yaml
model:
  name: "Qwen/Qwen2-7B"  # or Qwen/Qwen3-7B
  device: "cuda"
  dtype: "float16"
  use_cache: false

detection:
  prompt: "The quick brown fox jumps over the lazy dog"  # Simple prompt for detection
  max_length: 50
  target_layers: "all"  # or specify layer numbers

analysis:
  num_validation_prompts: 500
  datasets: ["wikitext2", "c4"]
  zero_shot_tasks: ["piqa", "arc_easy", "arc_challenge", "hellaswag"]

output:
  save_coordinates: true
  save_visualizations: true
  results_dir: "results"
```

---

### Phase 2: Model Loading (`src/model_loader.py`)

**Purpose**: Load Qwen models from Hugging Face with proper configuration.

**Key Functions**:
```python
def load_model(model_name, device="cuda", dtype=torch.float16):
    """
    Load model and tokenizer from HuggingFace
    
    Returns:
        model: The loaded model
        tokenizer: The tokenizer
        config: Model configuration
    """

def get_layer_structure(model):
    """
    Extract the transformer layer structure
    
    Returns:
        dict: {
            'num_layers': int,
            'layer_names': list,
            'mlp_structure': dict  # Information about MLP layers
        }
    """

def get_down_proj_weights(model, layer_idx):
    """
    Extract down_proj weight matrix from specific layer
    
    Args:
        model: The model
        layer_idx: Layer index
        
    Returns:
        torch.Tensor: Weight matrix [D, H]
    """
```

**Implementation Details**:
- Handle different model architectures (Qwen2/Qwen3 may have different layer naming)
- Support both full precision and half precision loading
- Implement device mapping for large models
- Extract MLP structure specifically for down_proj layers

---

### Phase 3: Activation Analysis (`src/activation_analyzer.py`)

**Purpose**: Capture and analyze activations during forward pass.

**Core Algorithm** (from paper Section 3.1):

The paper describes: For down_proj weight matrix W ∈ R^(D×H) and input X ∈ R^(L×H), output Y = XW^T. If Y_ij is a super activation and both X_ik and W_jk are outliers, then Y_ij ≈ X_ik * W_jk.

**Key Classes**:

```python
class ActivationCapture:
    """
    Hook-based activation capture during forward pass
    """
    def __init__(self, model, target_modules=['mlp.down_proj']):
        self.activations = {}
        self.hooks = []
        
    def register_hooks(self):
        """Register forward hooks on target modules"""
        
    def capture_forward_pass(self, input_ids):
        """
        Run forward pass and capture activations
        
        Returns:
            dict: {
                'layer_0': {
                    'input': tensor,
                    'output': tensor
                },
                ...
            }
        """
        
    def get_max_activations(self):
        """
        Get maximum activation values per layer
        
        Returns:
            dict: {
                'input_max': [(layer, channel, value), ...],
                'output_max': [(layer, channel, value), ...]
            }
        """
```

**Implementation Details**:
- Use PyTorch hooks to capture intermediate activations
- Store both input and output activations for down_proj
- Track maximum values and their indices
- Handle batch processing efficiently
- Clear hooks after use to prevent memory leaks

---

### Phase 4: Super Weight Detection (`src/super_weight_detector.py`)

**Purpose**: Implement the core detection algorithm from the paper.

**Detection Algorithm** (from paper Section 3.1):

1. Run single forward pass with a simple prompt
2. Plot extreme outliers in input/output of mlp.down_proj
3. Identify spikes in distributions
4. Map spike locations to weight coordinates
5. Remove detected super weight and repeat until no more spikes

**Key Functions**:

```python
class SuperWeightDetector:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.activation_capture = ActivationCapture(model)
        self.detected_weights = []
        
    def detect_all_super_weights(self, prompt, max_iterations=10):
        """
        Iteratively detect super weights
        
        Algorithm:
        1. Run forward pass and capture activations
        2. Find maximum activation spikes in down_proj input/output
        3. Determine super weight coordinates from spike locations
        4. Zero out the super weight
        5. Repeat until no significant spikes remain
        
        Returns:
            list: [(layer_num, row_idx, col_idx), ...]
        """
        
    def find_activation_spike(self, activations, threshold_percentile=99.9):
        """
        Identify activation spikes that indicate super weights
        
        From paper: Look for "extreme outliers" in down_proj input/output
        
        Returns:
            dict: {
                'layer': int,
                'input_channel': int,
                'output_channel': int,
                'input_magnitude': float,
                'output_magnitude': float
            }
        """
        
    def map_spike_to_weight_coordinates(self, spike_info):
        """
        Map activation spike to weight coordinate
        
        From paper Figure 3:
        - Input spike channel → weight row index
        - Output spike channel → weight column index
        
        Returns:
            tuple: (layer, row, col)
        """
        
    def zero_out_weight(self, layer, row, col):
        """Temporarily set weight to zero for iterative detection"""
        
    def validate_super_weight(self, layer, row, col):
        """
        Validate detected super weight by:
        1. Measuring activation change when removed
        2. Checking if it's in early layer (per paper)
        3. Verifying it's in down_proj
        
        Returns:
            bool: True if valid super weight
        """
```

**Implementation Details**:
- Handle numerical precision carefully (using float32 for comparisons)
- Implement proper spike detection with statistical thresholds
- Support iterative detection (paper mentions some models have up to 6)
- Validate each detection before adding to list
- Save intermediate results for debugging

---

### Phase 5: Visualization (`src/visualization.py`)

**Purpose**: Create publication-quality plots matching the paper's figures.

**Key Visualizations**:

```python
def plot_max_activations_per_layer(activations, title, save_path):
    """
    Recreate Figure 3 from paper:
    - Plot max activation value vs layer number
    - Separate plots for input and output of down_proj
    - Highlight spike locations
    """

def plot_activation_magnitude_distribution(activations, layer_idx):
    """
    Show distribution of activation magnitudes in a specific layer
    Helps identify outliers
    """

def plot_super_activation_persistence(model, super_weight_coords, prompts):
    """
    Recreate Figure 4 from paper:
    - Show how super activation persists across layers
    - Compare with/without super weight
    """

def visualize_token_probability_shift(logits_original, logits_no_sw, tokenizer):
    """
    Recreate Figure 5 from paper:
    - Show probability distribution of top tokens
    - Compare original vs. no super weight
    - Highlight stopword probability changes
    """

def create_comprehensive_report(detected_weights, validation_results):
    """
    Generate HTML report with all visualizations and metrics
    """
```

---

### Phase 6: Validation Experiments (`experiments/validate_super_weights.py`)

**Purpose**: Validate detected super weights following paper's methodology.

**Validation Tests**:

```python
def test_model_quality_degradation(model, tokenizer, super_weight_coords):
    """
    Test 1: Prune super weight and measure quality drop
    
    Metrics from paper:
    - Zero-shot accuracy on: ARC-c, ARC-e, HellaSwag, LAMBADA, PIQA, SciQ, Winogrande
    - Perplexity on: WikiText-2, C4
    
    Expected: Accuracy drops to ~35%, perplexity increases by 100-1000x
    """

def compare_super_weight_vs_top_outliers(model, tokenizer, super_weight_coords):
    """
    Test 2: Compare pruning super weight vs. pruning top 7000 outliers
    
    Expected: Super weight has much larger impact than 7000 other outliers
    """

def test_super_activation_creation(model, tokenizer, super_weight_coords):
    """
    Test 3: Verify super weight creates super activation
    
    Process:
    1. Run forward pass with super weight
    2. Measure activation at specific channel
    3. Prune super weight
    4. Run forward pass again
    5. Compare activation magnitude
    
    Expected: Activation magnitude drops by ~75% (from paper Figure 4)
    """

def test_stopword_probability_amplification(model, tokenizer, super_weight_coords):
    """
    Test 4: Check stopword probability changes
    
    Expected: Without super weight, stopwords ("the", ".", ",") increase 2-10x
    """

def test_sensitivity_to_scaling(model, tokenizer, super_weight_coords, scale_factors):
    """
    Test 5: Amplify super weight and measure accuracy change
    
    Scale factors: 0.0 to 3.0 (from paper Figure 6)
    Expected: Slight accuracy improvement with amplification
    """
```

---

### Phase 7: Main Detection Script (`experiments/detect_super_weights.py`)

**Purpose**: Orchestrate the complete detection pipeline.

```python
def main():
    # 1. Load configuration
    config = load_config("config/model_config.yaml")
    
    # 2. Load model
    model, tokenizer, model_config = load_model(
        config['model']['name'],
        device=config['model']['device'],
        dtype=config['model']['dtype']
    )
    
    # 3. Initialize detector
    detector = SuperWeightDetector(model, tokenizer, config)
    
    # 4. Run detection
    prompt = config['detection']['prompt']
    super_weights = detector.detect_all_super_weights(prompt)
    
    # 5. Validate detections
    validation_results = validate_super_weights(
        model, tokenizer, super_weights, config
    )
    
    # 6. Generate visualizations
    create_all_visualizations(
        detector.activation_capture.activations,
        super_weights,
        validation_results,
        config['output']['results_dir']
    )
    
    # 7. Save results
    save_results(super_weights, validation_results, config)
    
    # 8. Generate report
    generate_html_report(super_weights, validation_results)
```

---

## Step-by-Step Implementation Guide

### Step 1: Project Setup (Day 1)
1. Create directory structure
2. Set up virtual environment: `python -m venv venv`
3. Install dependencies: `pip install -r requirements.txt`
4. Create configuration file with Qwen model details
5. Initialize git repository

### Step 2: Model Loading (Day 1-2)
1. Implement `load_model()` function
2. Test loading Qwen2-7B or Qwen3-7B
3. Implement `get_layer_structure()` to understand architecture
4. Implement `get_down_proj_weights()` to access specific layers
5. Create unit tests for model loading

**Testing checkpoint**: Verify you can load model and access down_proj weights

### Step 3: Activation Capture (Day 2-3)
1. Implement `ActivationCapture` class with hooks
2. Test hook registration on a single layer
3. Implement full forward pass capture
4. Test with simple prompt and verify activations are captured
5. Implement `get_max_activations()` method

**Testing checkpoint**: Run forward pass and print max activation values per layer

### Step 4: Spike Detection (Day 3-4)
1. Implement `find_activation_spike()` with statistical threshold
2. Visualize activation distributions to tune threshold
3. Test spike detection on sample activations
4. Implement `map_spike_to_weight_coordinates()`

**Testing checkpoint**: Detect first spike and map to weight coordinates

### Step 5: Iterative Detection (Day 4-5)
1. Implement `zero_out_weight()` for temporary pruning
2. Implement iterative detection loop in `detect_all_super_weights()`
3. Add convergence criteria (no more significant spikes)
4. Test on full model

**Testing checkpoint**: Complete detection of all super weights in Qwen model

### Step 6: Visualization (Day 5-6)
1. Implement `plot_max_activations_per_layer()` (Figure 3 style)
2. Implement activation distribution plots
3. Create before/after comparison plots
4. Test all visualizations with detected results

**Testing checkpoint**: Generate all visualization plots

### Step 7: Validation Experiments (Day 6-8)
1. Implement perplexity calculation on WikiText-2 and C4
2. Integrate lm-evaluation-harness for zero-shot tasks
3. Implement quality degradation test
4. Implement super activation measurement test
5. Implement token probability analysis
6. Run all validation experiments

**Testing checkpoint**: Complete validation showing quality degradation

### Step 8: Analysis and Documentation (Day 8-9)
1. Analyze results and compare with paper findings
2. Create comprehensive report
3. Document any differences from paper
4. Create example notebooks showing the process
5. Write detailed README

---

## Key Implementation Notes

### Critical Details from Paper

1. **Super weights are always in early layers**: Focus detection on first 10 layers
2. **Always in `mlp.down_proj`**: Don't search in other layer types
3. **Single forward pass is sufficient**: No need for calibration data
4. **Detection is data-free**: Use any simple prompt
5. **Most models have ≤6 super weights**: Don't expect to find many

### Architecture-Specific Considerations

For Qwen2/Qwen3, you need to:
- Check the exact layer naming convention (might be `model.layers[i].mlp.down_proj` or similar)
- Verify if they use SwiGLU (paper mentions GLU variants create super activations)
- Check hidden dimensions and intermediate sizes
- Verify if they use RMSNorm or LayerNorm

### Performance Optimization

1. Use `torch.no_grad()` for activation capture (no backprop needed)
2. Use half precision (float16) to reduce memory
3. Clear CUDA cache between experiments
4. Use gradient checkpointing if needed for large models

### Common Pitfalls to Avoid

1. **Not clearing hooks**: Always remove hooks after use
2. **Precision issues**: Use float32 for statistical comparisons
3. **Index confusion**: Track tensor dimensions carefully (batch, seq, hidden)
4. **Memory leaks**: Delete large activation tensors when done
5. **Device mismatches**: Ensure all tensors on same device

---

## Expected Outputs

### 1. Super Weight Coordinates File
```json
{
  "model_name": "Qwen/Qwen2-7B",
  "detection_date": "2024-01-15",
  "super_weights": [
    {
      "layer": 2,
      "weight_type": "mlp.down_proj",
      "coordinates": [3968, 7003],
      "magnitude": 0.45,
      "validation": {
        "quality_drop": 0.52,
        "activation_reduction": 0.75
      }
    }
  ]
}
```

### 2. Validation Metrics
```json
{
  "original_model": {
    "avg_zero_shot_accuracy": 0.70,
    "wikitext2_perplexity": 5.68,
    "c4_perplexity": 7.08
  },
  "without_super_weight": {
    "avg_zero_shot_accuracy": 0.35,
    "wikitext2_perplexity": 763.65,
    "c4_perplexity": 1211.11
  }
}
```

### 3. Visualizations
- Activation spike plots (Figure 3 style)
- Super activation persistence (Figure 4 style)
- Token probability shifts (Figure 5 style)
- Sensitivity curves (Figure 6 style)

---

## Testing Strategy

### Unit Tests
- Test model loading with different configs
- Test activation capture on single layer
- Test spike detection with synthetic data
- Test coordinate mapping

### Integration Tests
- Test complete detection pipeline
- Test validation experiments
- Test visualization generation

### Validation Tests
- Compare results with paper's findings for similar models
- Verify quality degradation matches expected pattern
- Check super activation behavior matches paper

---

## Extensions and Future Work

1. **Quantization experiments**: Implement super-weight-aware quantization
2. **Multi-model comparison**: Detect super weights across model families
3. **Fine-tuning analysis**: Check if super weights change during fine-tuning
4. **Amplification experiments**: Test if amplifying super weights improves quality
5. **Transfer learning**: Study if super weights transfer across tasks

---

## Resources and References

- Paper: "The Super Weight in Large Language Models" (ICLR 2025 submission)
- Hugging Face Transformers: https://huggingface.co/docs/transformers
- lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness
- PyTorch hooks tutorial: https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html

---

## Success Criteria

The implementation is successful if:
1. ✅ Successfully detect super weights in Qwen model
2. ✅ Reproduce paper's key finding: pruning super weight destroys quality
3. ✅ Generate all visualizations matching paper's figures
4. ✅ Document super weight coordinates for Qwen models
5. ✅ Create reusable detection pipeline for other models