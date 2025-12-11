# Installation

Create a Conda environment:

```bash
conda env create -f conda_env.yml
conda activate spin
```

Requirements:

- Python 3.8
- PyTorch >= 1.9
- PyTorch Geometric >= 2.0
- torch_spatiotemporal 0.1.1

## Training

Train the SPIN model with predefined configuration:

```bash
python run_imputation.py \
    --model-name spin \
    --dataset-name air36 \
    --config imputation/spin.yaml
```

**Training Parameters:**

- `--model-name`: Model type (`spin`, `spin_h`, `brits`, `saits`, `grin`, `transformer`)
- `--dataset-name`: Dataset name (`air36`, `air`, `la_point`, `bay_point`, `la_block`, `bay_block`)
- `--config`: Configuration file path
- `--seed`: Random seed (default: -1)
- `--epochs`: Number of epochs (default: 300)
- `--batch-size`: Batch size (default: 8)
- `--lr`: Learning rate (default: 0.0008)
- `--window`: Time window size (default: 24)

## Inference

Run inference with a trained model:

```bash
python run_inference.py \
    --model-name spin \
    --dataset-name air36 \
    --exp-name my_experiment \
    --root log
```

**Inference Parameters:**

- `--exp-name`: Experiment name (required)
- `--root`: Log root directory (default: log)
- `--p-fault`: Fault probability (default: 0.0)
- `--p-noise`: Noise probability (default: 0.75)
- `--test-mask-seed`: Test mask random seed
- `--batch-size`: Batch size (default: 32)
- `--adj-threshold`: Adjacency matrix threshold (default: 0.1)

## Attribution Analysis

Generate model attributions:

```bash
python generate_attributions.py \
    --model-name spin \
    --dataset-name air36 \
    --exp-name my_experiment \
    --epsilon 0.01 \
    --num-steps 10
```

**Attribution Parameters:**

- `--epsilon`: Perturbation magnitude (default: 0.01)
- `--num-steps`: Number of iterations (default: 10)
- `--alpha`: Step size (default: 0.001)
- `--out-dir`: Output directory (default: attributions)
