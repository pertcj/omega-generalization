# Installation

```bash
conda create --name omega python=3.10
conda activate omega

# If pip wasn't already installed 
conda install pip

# Install Spot (for LTL to automaton conversion)
# This means it must be installed first from: https://spot.lre.epita.fr/install.html
# Then the bindings must be installed to the environment with the following command.
conda install conda-forge::spot

# Installing the package should install the required dependencies automatically
pip install -e .

# (Optional) Upgrade JAX version to GPU-ready version
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

# Usage

## Available Tasks

The following LTL formulas are already implemented:

- Lift Formulas:
    - `lift_X[a/b]`: where $X \in \{2, \dots, 8\}$ is the number of floors in the lift system. The `a` or `b` correspond to the different formula encodings of the same lift behavior. 
- Acacia Formulas:
    - `neg_formula_X`: where $X \in \{1, \dots, 25\}$ corresponds to the specific LTL formula from the Acacia benchmark.

Edit `omega_generalization/training/constants.py` to include additional formulas. If the automaton produced by Spot is nondeterministic, the script will throw an error.

## Parameters

- `--task`: Task name (required)
- `--benchmark`: Benchmark name (required; choices=[lift, acacia])
- `--steps`: Number of training steps (default: 1,000,000)
- `--hidden_dim`: Model hidden dimension (default: 256)
- `--batch_size`: Training batch size (default: 256)  
- `--sequence_length`: Training sequence length (default: 64)
- `--lr`: Learning rate (default: 1e-3)
- `--seed`: Random seed (default: 0)
- `--max_range_test_length`: Maximum length for generalization testing (default: 512)
- `--use_tensorboard`: Enable TensorBoard logging (default: False)

## Example

```bash
cd omega_generalization
python training/example.py --task neg_formula_1 --benchmark acacia --steps 10000 --hidden_dim 64 --batch_size 32 --sequence_length 8 --lr 1e-3 --seed 0 --max_range_test_length 32
```

## Output

The script will output training progress and final evaluation results. The in-distribution (ID) accuracy is printed at the end of training which is the average accuracy on sampled sequences between length `2` to `sequence_length`. The out-of-distribution (OOD) accuracy is also printed at the end of training, which is the average accuracy on sequences longer than the training length (from `sequence_length`+1 to `max_range_test_length`). 

# Code Reference
This repository builds upon the implementation from [https://github.com/google-deepmind/neural_networks_chomsky_hierarchy](https://github.com/google-deepmind/neural_networks_chomsky_hierarchy)