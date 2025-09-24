# Contributing to MACE Model Deviation

## Development Setup

```bash
git clone https://github.com/lianxiliang/mace-model-deviation.git
cd mace-model-deviation
pip install -e ".[dev]"
```

## Testing

```bash
# Test with sample data
mace-model-devi --models model1.pt model2.pt --traj trajectory.xyz --output test.out

# Test Python API
python -c "from mace_model_deviation import calculate_mace_model_deviation; print('Import successful')"
```

## Integration with ai2-kit

After installing the package, ai2-kit will automatically use it for MACE model deviation calculations in SLURM environments.

## Docker Integration

Add to your Dockerfile:
```dockerfile
RUN pip install git+https://github.com/lianxiliang/mace-model-deviation.git
```

## Release Process

1. Update version in `pyproject.toml` and `setup.py`
2. Create git tag: `git tag v0.1.0`
3. Push tag: `git push origin v0.1.0`
4. GitHub Actions will handle the rest (if configured)