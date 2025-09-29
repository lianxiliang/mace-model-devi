# MACE Model Deviation Calculator

A standalone Python package for calculating ensemble uncertainty from MACE (Multi-Atomic Cluster Expansion) models.

## Features

- **Standalone**: Independent package with minimal dependencies
- **Fast**: Optimized for GPU calculation with CUDA support
- **Flexible**: Works with any MACE models and trajectory formats
- **Memory Efficient**: Uses MACE's native DataLoader for optimal batching
- **Easy Integration**: Simple pip install for any project
- **DeepMD Compatible**: Outputs standard model deviation format

## Installation

```bash
# Install from GitHub
pip install git+https://github.com/lianxiliang/mace-model-devi.git

# Or install from source
git clone https://github.com/lianxiliang/mace-model-devi.git
cd mace-model-devi
pip install -e .

# For development
pip install -e .[dev]
```

## Usage

### Command Line Interface

```bash
# Basic usage
mace-model-devi --models model1.model model2.model model3.model \
                --traj trajectory.xyz \
                --output model_devi.out

# With GPU and type mapping
mace-model-devi --models *.model \
                --traj traj.lammpstrj \
                --output results.out \
                --device cuda \
                --type-map O,H

# With batch size optimization
mace-model-devi --models *.model \
                --traj trajectory.xyz \
                --output model_devi.out \
                --batch-size 32 \
                --device cpu
```

### Python API

```python
from mace_model_deviation import calculate_mace_model_deviation

# Calculate model deviation
output_file = calculate_mace_model_deviation(
    model_files=['model1.model', 'model2.model', 'model3.model'],
    trajectory_file='trajectory.xyz',
    output_file='model_devi.out',
    device='cuda',
    batch_size=64
)
```

## Integration with ai2-kit

The package integrates seamlessly with ai2-kit workflows:

```python
# In ai2-kit macelmp.py
mace_cmd = f'mace-model-devi --models "{models}" --traj traj.lammpstrj --output model_devi.out --device cuda'
```

## Output Format

The output file follows DeepMD model deviation format with energy and force deviations:

```
#        step     max_devi_v     min_devi_v     avg_devi_v     max_devi_f     min_devi_f     avg_devi_f
           0   3.184592e-03   3.184592e-03   3.184592e-03   1.132804e-02   8.073725e-03   1.014817e-02
           1   3.442520e-03   3.442520e-03   3.442520e-03   1.424466e-02   4.970430e-03   1.107935e-02
           2   3.697712e-03   3.697712e-03   3.697712e-03   1.214610e-02   6.534542e-03   8.718190e-03
```

Where:
- `max/min/avg_devi_v`: Energy deviation statistics (used as virial proxy)
- `max/min/avg_devi_f`: Force deviation statistics per atom

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.11.0
- ASE ≥ 3.22.0
- NumPy ≥ 1.21.0
- MACE-torch ≥ 0.3.0

## Container Usage

Perfect for Docker containers and HPC environments:

```dockerfile
# In your Dockerfile
RUN pip install mace-model-deviation

# Or for development
COPY mace-model-deviation /opt/mace-model-deviation
RUN pip install -e /opt/mace-model-deviation
```

## Authors

- **Xiliang LIAN** - *Initial development and MACE integration*

## Credits

The work is based on the original mace eval_configs.py, which evaluates the configurations

## License

MIT License - see LICENSE file for details.