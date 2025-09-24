# MACE Model Deviation Calculator

A standalone Python package for calculating ensemble uncertainty from MACE (Multi-Atomic Cluster Expansion) models.

## Features

- **Standalone**: No dependency on ai2-kit or other frameworks
- **Fast**: Optimized for GPU calculation with CUDA support
- **Flexible**: Works with any MACE models and trajectory formats
- **Memory Efficient**: Processes trajectories in chunks
- **Easy Integration**: Simple pip install for any project

## Installation

```bash
# Install from source
pip install -e .

# Or install dependencies manually
pip install torch ase numpy
```

## Usage

### Command Line Interface

```bash
# Basic usage
mace-model-devi --models model1.pt model2.pt model3.pt \
                --traj trajectory.xyz \
                --output model_devi.out

# With GPU and type mapping
mace-model-devi --models *.pt \
                --traj traj.lammpstrj \
                --output results.out \
                --device cuda \
                --type-map O,H
```

### Python API

```python
from mace_model_deviation import calculate_mace_model_deviation

# Calculate model deviation
output_file = calculate_mace_model_deviation(
    model_files=['model1.pt', 'model2.pt', 'model3.pt'],
    trajectory_file='trajectory.xyz',
    output_file='model_devi.out',
    device='cuda'
)
```

## Integration with ai2-kit

The package integrates seamlessly with ai2-kit workflows:

```python
# In ai2-kit macelmp.py
mace_cmd = f'mace-model-devi --models "{models}" --traj traj.lammpstrj --output model_devi.out --device cuda'
```

## Output Format

The output file contains frame-by-frame maximum force deviations:

```
# Frame  Max_Force_Devi
     0      0.123456
     1      0.234567
     2      0.345678
```

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.11.0
- ASE ≥ 3.22.0
- NumPy ≥ 1.21.0

## Container Usage

Perfect for Docker containers and HPC environments:

```dockerfile
# In your Dockerfile
RUN pip install mace-model-deviation

# Or for development
COPY mace-model-deviation /opt/mace-model-deviation
RUN pip install -e /opt/mace-model-deviation
```

## License

MIT License - see LICENSE file for details.